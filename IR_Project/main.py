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
                return json.load(f)
        except FileNotFoundError:
            return {
                'last_processed_chunk': 0,
                'completed_embeddings': [],
                'last_update': str(datetime.now()),
                'total_chunks': 0,
                'processed_pdfs': []
            }
    
    def save_state(self):
        self.state['last_update'] = str(datetime.now())
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
    
    def update_progress(self, chunk_index: int, new_embeddings: List):
        self.state['last_processed_chunk'] = chunk_index
        self.state['completed_embeddings'].extend(new_embeddings)
        self.save_state()
    
    def add_processed_pdf(self, pdf_path: str):
        if pdf_path not in self.state['processed_pdfs']:
            self.state['processed_pdfs'].append(pdf_path)
            self.save_state()

def extract_content_from_pdf(pdf_path: str) -> str:
    """Extract text and tables from PDF"""
    full_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract regular text
            text = page.extract_text() or ""
            if text:
                full_content.append(text)
            
            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                df = df.fillna('')
                df = df.replace(r'^\s*$', '', regex=True)
                
                table_text = "\nTable Content:\n"
                table_text += df.to_string(index=False, header=False) + "\n"
                full_content.append(table_text)
    
    return "\n".join(full_content)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nTable Content:\n", "\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def check_api_quota(api_key: str) -> bool:
    """Check if API key has available quota"""
    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            input=["test"],
            model="text-embedding-ada-002"
        )
        return True
    except Exception as e:
        if "insufficient_quota" in str(e):
            return False
        return True

def get_embeddings_with_enhanced_retry(
    chunks: List[str],
    state: ProcessingState,
    model: str = "text-embedding-ada-002",
    api_key: str = None,
    max_retries: int = 3,
    initial_retry_delay: int = 20,
    batch_size: int = 50
) -> List[List[float]]:
    """Get embeddings with retry logic"""
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    embeddings = state.state['completed_embeddings']
    start_idx = state.state['last_processed_chunk']
    
    print(f"Resuming from chunk {start_idx} of {len(chunks)}")
    
    for i in range(start_idx, len(chunks), batch_size):
        if not check_api_quota(api_key):
            print("\nAPI quota exhausted. Saving state and waiting for quota renewal.")
            print(f"Progress: {i}/{len(chunks)} chunks processed")
            state.update_progress(i, embeddings)
            raise Exception("API quota exhausted. Please update API key or wait for quota renewal.")
        
        batch = chunks[i:i + batch_size]
        retry_delay = initial_retry_delay
        
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(input=batch, model=model)
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                state.update_progress(i + batch_size, batch_embeddings)
                print(f"Processed chunks {i} to {min(i + batch_size, len(chunks))}")
                time.sleep(1)
                break
                
            except Exception as e:
                error_msg = str(e)
                if "insufficient_quota" in error_msg:
                    print(f"\nQuota exceeded at chunk {i}. Progress saved.")
                    state.update_progress(i, embeddings)
                    raise Exception("API quota exhausted. Please update API key or wait for quota renewal.")
                
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    state.update_progress(i, embeddings)
                    raise Exception(f"Failed after {max_retries} attempts. Last error: {e}")
    
    return embeddings

def store_in_faiss(embeddings: List[List[float]], chunks: List[str], 
                   index_path: str = "vector_index.faiss", 
                   metadata_path: str = "chunks_metadata.pkl") -> None:
    """Store embeddings in FAISS and save metadata"""
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create and save FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    faiss.write_index(index, index_path)
    
    # Save chunks metadata
    pd.DataFrame({'chunk_text': chunks}).to_pickle(metadata_path)
    print(f"Index saved to {index_path}")
    print(f"Chunks metadata saved to {metadata_path}")

def verify_faiss_storage(index_path: str = "vector_index.faiss", 
                        metadata_path: str = "chunks_metadata.pkl") -> Tuple[int, int]:
    """Verify FAISS storage and return counts"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    index = faiss.read_index(index_path)
    chunks_df = pd.read_pickle(metadata_path)
    return index.ntotal, len(chunks_df)

def test_pipeline(pdf_folder: str, api_key: str):
    """Main pipeline function"""
    state = ProcessingState()
    
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
    
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) 
                 if f.endswith('.pdf') and 
                 os.path.join(pdf_folder, f) not in state.state['processed_pdfs']]
    
    if not pdf_files and not state.state['completed_embeddings']:
        raise ValueError(f"No new PDF files found in {pdf_folder}")
    
    all_chunks = []
    
    for pdf in pdf_files:
        try:
            print(f"\nProcessing: {pdf}")
            content = extract_content_from_pdf(pdf)
            chunks = chunk_text(content)
            all_chunks.extend(chunks)
            state.add_processed_pdf(pdf)
            print(f"Chunks created from {pdf}: {len(chunks)}")
            print("\nSample of extracted content:")
            print(content[:500] + "...")
        except Exception as e:
            print(f"Error processing {pdf}: {e}")
            continue
    
    state.state['total_chunks'] = len(all_chunks)
    state.save_state()
    
    print(f"\nTotal chunks to process: {len(all_chunks)}")
    
    try:
        embeddings = get_embeddings_with_enhanced_retry(all_chunks, state, api_key=api_key)
        store_in_faiss(embeddings, all_chunks)
        
        num_vectors, num_chunks = verify_faiss_storage()
        print(f"\nProcessing completed successfully!")
        print(f"Total vectors stored in FAISS: {num_vectors}")
        print(f"Total chunks stored in metadata: {num_chunks}")
        
        if os.path.exists(state.state_file):
            os.remove(state.state_file)
            
    except Exception as e:
        print(f"\nProcessing paused: {e}")
        print("Current progress has been saved. You can resume by running the script again with a valid API key.")
        print(f"Processed {state.state['last_processed_chunk']} out of {state.state['total_chunks']} total chunks")

if __name__ == "__main__":
    pdf_folder = "./pdfs"
    openai_api_key = "Add you OpenAI API key"  # Replace with your actual OpenAI API key
    
    try:
        test_pipeline(pdf_folder, openai_api_key)
    except Exception as e:
        print(f"Pipeline execution paused: {e}")