import os
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import pdfplumber
import pandas as pd
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ProcessingState:
    def __init__(self, state_file="processing_state.json"):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self) -> Dict:
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                # Ensure all required keys exist
                default_state = {
                    'all_chunks': [],
                    'processed_pdfs': [],
                    'completed_embeddings': [],
                    'last_processed_chunk': 0,
                    'last_update': str(datetime.now())
                }
                return {**default_state, **state}
        except FileNotFoundError:
            return {
                'all_chunks': [],
                'processed_pdfs': [],
                'completed_embeddings': [],
                'last_processed_chunk': 0,
                'last_update': str(datetime.now())
            }
    
    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
    
    def reset_embeddings(self):
        """Reset embeddings while keeping other state intact"""
        self.state['completed_embeddings'] = []
        self.state['last_processed_chunk'] = 0
        self.save_state()
    
    def update_progress(self, chunk_index: int, new_embeddings: List):
        self.state['last_processed_chunk'] = chunk_index
        self.state['completed_embeddings'] = new_embeddings  # Store only current embeddings
        self.save_state()
    
    def add_processed_pdf(self, pdf_path: str, chunks: List[str]):
        if pdf_path not in self.state['processed_pdfs']:
            self.state['processed_pdfs'].append(pdf_path)
            self.state['all_chunks'].extend(chunks)
            self.save_state()

def extract_content_from_pdf(pdf_path: str) -> str:
    """Extract text and tables from PDF with improved error handling"""
    full_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text with error handling
                    text = page.extract_text() or ""
                    if text:
                        full_content.append(text)
                    
                    # Extract tables with error handling
                    tables = page.extract_tables()
                    for table in tables:
                        if table:  # Check if table is not empty
                            df = pd.DataFrame(table).fillna('').replace(r'^\s*$', '', regex=True)
                            full_content.append(f"\nTable Content:\n{df.to_string(index=False, header=False)}\n")
                except Exception as e:
                    print(f"Warning: Error processing page {page_num} in {pdf_path}: {str(e)}")
                    continue
                
    except Exception as e:
        print(f"Error: Failed to process PDF {pdf_path}: {str(e)}")
        return ""  # Return empty string on complete failure
    
    return "\n".join(full_content)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks with empty text handling"""
    if not text.strip():
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nTable Content:\n", "\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def get_embeddings_with_enhanced_retry(
    chunks: List[str],
    state: ProcessingState,
    model: str = "text-embedding-ada-002",
    api_key: str = None,
    max_retries: int = 3,
    initial_retry_delay: int = 20,
    batch_size: int = 50
) -> List[List[float]]:
    """Get embeddings with improved state management"""
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    embeddings = []  # Start fresh
    start_idx = 0  # Always start from beginning
    
    print(f"Processing {len(chunks)} chunks")
    
    for i in range(start_idx, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        retry_delay = initial_retry_delay
        
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(input=batch, model=model)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Update state with current progress
                state.update_progress(i + len(batch), embeddings)
                print(f"Processed chunks {i}-{i+len(batch)-1}")
                time.sleep(1)
                break
            except Exception as e:
                if "insufficient_quota" in str(e):
                    raise RuntimeError("API quota exhausted") from e
                if attempt == max_retries - 1:
                    raise
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
    
    return embeddings



def verify_faiss_storage(index_path: str = "vector_index.faiss", 
                        metadata_path: str = "chunks_metadata.pkl") -> Tuple[int, int]:
    """
    Verify the integrity of stored FAISS index and metadata.
    
    This function performs several important checks:
    1. Verifies that both index and metadata files exist
    2. Ensures vector count matches chunk count
    3. Validates embedding dimensions
    4. Checks metadata structure
    
    Args:
        index_path: Path to the FAISS index file
        metadata_path: Path to the metadata pickle file
    
    Returns:
        Tuple containing (number of vectors, number of chunks)
    
    Raises:
        FileNotFoundError: If either file is missing
        ValueError: If there's a mismatch or validation fails
    """
    # Check if files exist
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load FAISS index and metadata
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        raise ValueError(f"Failed to load FAISS index: {str(e)}")
    
    try:
        metadata = pd.read_pickle(metadata_path)
    except Exception as e:
        raise ValueError(f"Failed to load metadata: {str(e)}")
    
    # Verify index has vectors
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty")
    
    # Verify metadata structure
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("Metadata must be a pandas DataFrame")
    if 'chunk_text' not in metadata.columns:
        raise ValueError("Metadata missing 'chunk_text' column")
    
    # Verify counts match
    if index.ntotal != len(metadata):
        raise ValueError(
            f"Count mismatch: FAISS index has {index.ntotal} vectors, "
            f"but metadata has {len(metadata)} chunks"
        )
    
    # Verify embedding dimensions
    try:
        sample_embedding = index.reconstruct(0)  # Get first vector
        if len(sample_embedding) != 1536:  # OpenAI ada-002 dimension
            raise ValueError(
                f"Unexpected embedding dimension: {len(sample_embedding)}. "
                f"Expected 1536 for ada-002 model."
            )
    except Exception as e:
        raise ValueError(f"Failed to verify embedding dimensions: {str(e)}")
    
    print(f"Storage verification successful:")
    print(f"- FAISS index contains {index.ntotal} vectors")
    print(f"- Metadata contains {len(metadata)} chunks")
    print(f"- Embedding dimension: {len(sample_embedding)}")
    
    return index.ntotal, len(metadata)

def store_in_faiss(embeddings: List[List[float]], chunks: List[str], 
                   index_path: str = "vector_index.faiss", 
                   metadata_path: str = "chunks_metadata.pkl") -> None:
    """
    Store embeddings and chunks in FAISS index and metadata file.
    
    This function ensures data consistency by:
    1. Validating input data
    2. Creating a new FAISS index
    3. Storing chunks metadata
    4. Verifying the stored data
    
    Args:
        embeddings: List of embedding vectors
        chunks: List of text chunks
        index_path: Path to save FAISS index
        metadata_path: Path to save metadata
    """
    # Input validation
    if not embeddings or not chunks:
        raise ValueError("No embeddings or chunks to store")
    
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Count mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
        )
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Validate embedding dimensions
    if embeddings_array.shape[1] != 1536:  # OpenAI ada-002 dimension
        raise ValueError(
            f"Unexpected embedding dimension: {embeddings_array.shape[1]}"
        )
    
    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    faiss.write_index(index, index_path)
    
    # Save metadata
    metadata = pd.DataFrame({
        'chunk_text': chunks,
        'embedding_index': range(len(chunks))  # Add index for reference
    })
    metadata.to_pickle(metadata_path)
    
    # Verify storage
    try:
        num_vectors, num_chunks = verify_faiss_storage(index_path, metadata_path)
        print(f"Successfully stored {num_vectors} vectors with metadata")
    except Exception as e:
        # Clean up on verification failure
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        raise ValueError(f"Storage verification failed: {str(e)}")

# [Rest of the code remains the same]

def test_pipeline(pdf_folder: str, api_key: str):
    """Main pipeline with improved error handling"""
    state = ProcessingState()
    
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
    
    # Clear previous embeddings to avoid accumulation
    state.reset_embeddings()
    
    # Process new PDFs
    new_pdfs = [
        os.path.join(pdf_folder, f) 
        for f in os.listdir(pdf_folder) 
        if f.endswith('.pdf') and 
        os.path.join(pdf_folder, f) not in state.state['processed_pdfs']
    ]
    
    for pdf in new_pdfs:
        try:
            print(f"\nProcessing {pdf}")
            content = extract_content_from_pdf(pdf)
            if not content:
                print(f"Warning: No content extracted from {pdf}")
                continue
                
            chunks = chunk_text(content)
            if chunks:
                state.add_processed_pdf(pdf, chunks)
                print(f"Added {len(chunks)} chunks from {os.path.basename(pdf)}")
            else:
                print(f"Warning: No chunks created from {pdf}")
        except Exception as e:
            print(f"Error processing {pdf}: {str(e)}")
            continue
    
    # Get all chunks and process
    all_chunks = state.state['all_chunks']
    if not all_chunks:
        print("No chunks to process. Check PDF content and extraction.")
        return
    
    print(f"\nTotal chunks to process: {len(all_chunks)}")
    
    try:
        embeddings = get_embeddings_with_enhanced_retry(all_chunks, state, api_key=api_key)
        store_in_faiss(embeddings, all_chunks)
        
        # Verify storage
        num_vectors, num_chunks = verify_faiss_storage()
        print(f"\nSuccess: {num_vectors} vectors stored for {num_chunks} chunks")
        
        # Clean up state file only on complete success
        if os.path.exists(state.state_file):
            os.remove(state.state_file)
            
    except Exception as e:
        print(f"\nProcessing paused: {str(e)}")
        print("Current progress saved. You can resume by running the script again.")

if __name__ == "__main__":
    pdf_folder = "./pdfs"
    openai_api_key = "your-api-key"  # Replace with actual key
    
    try:
        test_pipeline(pdf_folder, openai_api_key)
    except Exception as e:
        print(f"Critical error: {str(e)}")