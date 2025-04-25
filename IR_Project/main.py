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
                default_state = {
                    'all_chunks': [],
                    'processed_pdfs': [],
                    'completed_embeddings': [], # completed embeddings are the embeddings that have been successfully processed and stored in the FAISS index. example: if you have 100 chunks and you have processed 50 of them, then the completed_embeddings list will contain the embeddings for those 50 chunks.
                    'last_processed_chunk': 0,
                    'last_update': str(datetime.now())
                }
                return {**default_state, **state} # the **  operator is used for dictionary unpacking, it takes all key value pairs.  and merges default_state and state dictionaries. with state values overriding the ones in default_state if they share the same keys.
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
        self.state['completed_embeddings'] = []
        self.state['last_processed_chunk'] = 0
        self.save_state()
    
    def update_progress(self, chunk_index: int, new_embeddings: List):
        self.state['last_processed_chunk'] = chunk_index
        self.state['completed_embeddings'] = new_embeddings
        self.save_state()
    
    def add_processed_pdf(self, pdf_path: str, chunks: List[str]):
        if pdf_path not in self.state['processed_pdfs']:
            self.state['processed_pdfs'].append(pdf_path)
            self.state['all_chunks'].extend(chunks)
            self.save_state()

def extract_content_from_pdf(pdf_path: str) -> str: # here... could extract bullet points separately from text and tables, and append separately to table
    full_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    full_content.append(text)
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table).fillna('').replace(r'^\s*$', '', regex=True) # remove empty strings 
                        full_content.append(f"\nTable Content:\n{df.to_string(index=False, header=False)}\n")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""
    return "\n".join(full_content) # this will join all the text and tables extracted from the pdf into a single string. The tables are formatted as strings with headers and indices removed. 

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]: #ISSUE: IF TABLE EXCEEDS CHUNK SIZE, IT WILL CUT THE TABLE SHORT
    if not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nTable Content:\n", "\n\n", "\n", " ", ""]
    ) # this is a list of separators that will be used to split the text into chunks. The first one is the most important, as it will be used to split the text into chunks based on the table content. The rest are just for splitting the text into smaller chunks.

    return text_splitter.split_text(text) # this will split the text into chunks of size chunk_size, with an overlap of chunk_overlap characters between consecutive chunks.

def get_embeddings_with_enhanced_retry(
    chunks: List[str],
    state: ProcessingState,
    model: str = "text-embedding-ada-002",
    api_key: str = None,
    max_retries: int = 3,
    initial_retry_delay: int = 20,
    batch_size: int = 50
) -> List[List[float]]:
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = OpenAI(api_key=api_key)
    embeddings = []
    start_idx = 0
    
    print(f"Processing {len(chunks)} chunks")
    
    for i in range(start_idx, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        retry_delay = initial_retry_delay
        
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(input=batch, model=model) # PASSES THE BATCH OF CHUNKS TO THE OPENAI API FOR EMBEDDING
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                state.update_progress(i + len(batch), embeddings)
                print(f"Processed chunks {i}-{i+len(batch)-1}")
                time.sleep(1)
                break
            except Exception as e:
                if "insufficient_quota" in str(e):
                    raise RuntimeError("API quota exhausted") from e
                if attempt == max_retries - 1:
                    raise # THIS MANUALLY RAISES AN EXCEPTION / ERROR
                print(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
    
    return embeddings

def verify_faiss_storage(index_path: str = "vector_index4.faiss", metadata_path: str = "chunks_metadata4.pkl") -> Tuple[int, int]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        raise ValueError(f"Failed to load FAISS index: {str(e)}")
    
    try:
        metadata = pd.read_pickle(metadata_path)
    except Exception as e:
        raise ValueError(f"Failed to load metadata: {str(e)}")
    
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty")
    
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("Metadata must be a pandas DataFrame")
    if 'chunk_text' not in metadata.columns:
        raise ValueError("Metadata missing 'chunk_text' column")
    
    if index.ntotal != len(metadata):
        raise ValueError(f"Count mismatch: FAISS index has {index.ntotal} vectors, but metadata has {len(metadata)} chunks")
    
    try:
        sample_embedding = index.reconstruct(0)
        if len(sample_embedding) != 1536:
            raise ValueError(f"Unexpected embedding dimension: {len(sample_embedding)}. Expected 1536 for ada-002 model.")
    except Exception as e:
        raise ValueError(f"Failed to verify embedding dimensions: {str(e)}")
    
    print(f"Storage verification successful:")
    print(f"- FAISS index contains {index.ntotal} vectors")
    print(f"- Metadata contains {len(metadata)} chunks")
    print(f"- Embedding dimension: {len(sample_embedding)}")
    
    return index.ntotal, len(metadata)

def store_in_faiss(embeddings: List[List[float]], chunks: List[str], index_path: str = "vector_index4.faiss", metadata_path: str = "chunks_metadata4.pkl") -> None:
    if not embeddings or not chunks:
        raise ValueError("No embeddings or chunks to store")
    
    if len(embeddings) != len(chunks):
        raise ValueError(f"Count mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")
    
    embeddings_array = np.array(embeddings).astype('float32')

    # removed in case a model other than ada is used    
    # if embeddings_array.shape[1] != 1536:
    #     raise ValueError(f"Unexpected embedding dimension: {embeddings_array.shape[1]}")
    
    index = faiss.IndexFlatL2(embeddings_array.shape[1]) # stores the embeddings in an  "index" which abstracts away the details of the underlying data structure. The IndexFlatL2 is a simple index that uses L2 distance (Euclidean distance) for nearest neighbor search.
    index.add(embeddings_array)
    faiss.write_index(index, index_path)
    
    metadata = pd.DataFrame({ # metadata is a pandas dataframe that stores the chunks and their corresponding indices. This is used to retrieve the chunks later when querying the index.
        'chunk_text': chunks,
        'embedding_index': range(len(chunks))
    })
    metadata.to_pickle(metadata_path) # pickle basically helps to serialize the dataframe into a binary format that can be saved to disk and loaded back later. (efficient retreival) could also use json, but pickle is faster and more efficient for large dataframes.
    
    try:
        num_vectors, num_chunks = verify_faiss_storage(index_path, metadata_path) # this will verify that the vectors and metadata have been stored correctly in the FAISS index and the metadata file.
        print(f"Successfully stored {num_vectors} vectors with metadata")
    except Exception as e:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        raise ValueError(f"Storage verification failed: {str(e)}")

def test_pipeline(pdf_folder: str, api_key: str):
    state = ProcessingState()
    
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")
    
    state.reset_embeddings()
    
    new_pdfs = [
        os.path.join(pdf_folder, f) 
        for f in os.listdir(pdf_folder) 
        if f.endswith('.pdf') and 
        os.path.join(pdf_folder, f) not in state.state['processed_pdfs']
    ] # this will get all the pdfs in the folder that have not been processed yet.
    
    for pdf in new_pdfs: # THIS IS THE MAIN LOOP, IT EXTRACTS THE CONTENT FROM THE PDF AND CHUNKS IT INTO SMALLER PIECES. THEN IT UPDATES THE STATE WITH THE NEW CHUNKS. (CHUNKING DOES NOT MEAN EMBEDDING, IT JUST MEANS SPLITTING THE TEXT INTO SMALLER PIECES) (HENCE CHUNKING CAN BE IMPROVED)
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
    
    all_chunks = state.state['all_chunks']
    if not all_chunks:
        print("No chunks to process. Check PDF content and extraction.")
        return
    
    print(f"\nTotal chunks to process: {len(all_chunks)}")
    
    try:
        embeddings = get_embeddings_with_enhanced_retry(all_chunks, state, api_key=api_key) # this will get the embeddings for all the chunks. It will retry if it fails, and it will update the state with the new embeddings.
        store_in_faiss(embeddings, all_chunks) # this will store the embeddings in the FAISS index and the metadata file.
        
        num_vectors, num_chunks = verify_faiss_storage()
        print(f"\nSuccess: {num_vectors} vectors stored for {num_chunks} chunks")
        
        if os.path.exists(state.state_file):
            os.remove(state.state_file)
            
    except Exception as e:
        print(f"\nProcessing paused: {str(e)}")
        print("Current progress saved. You can resume by running the script again.")

if __name__ == "__main__":
    pdf_folder = "./pdfs"
    openai_api_key = ""  
    print("A message: Open AI API KEY Needed")
    try:
        test_pipeline(pdf_folder, openai_api_key)
    except Exception as e:
        print(f"Critical error: {str(e)}")