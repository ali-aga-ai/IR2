import faiss
import numpy as np
import openai
import pickle
from openai import OpenAI


def get_embedding(text, api_key,model="text-embedding-ada-002"):

    client = OpenAI(api_key=api_key)
    text = text.replace("\n", " ")

    return client.embeddings.create(input = [text], model=model).data[0].embedding

def query_faiss(query_text, api_key, top_k=4):

    index = faiss.read_index("vector_index.faiss")

    with open("chunks_metadata.pkl", "rb") as f:
        pdf_files = pickle.load(f)

    query_vector = get_embedding(query_text, api_key)
    query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
    
    #print(f"DataFrame size: {len(pdf_files)}")
    
    distances, indices = index.search(query_vector, top_k)
    
    #print(f"FAISS returned indices: {indices[0]}")
    
    # Add bounds checking
    valid_results = []

    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(pdf_files):
            valid_results.append((pdf_files.iloc[idx]['chunk_text'], distances[0][i]))
            print(pdf_files.iloc[idx]['pdf_files'])
        else:
            print(f"Warning: Index {idx} is out of bounds")
    
    return valid_results


def query(userQuery, openai_api_key):
    results = query_faiss(userQuery,openai_api_key)
    # print("\nResults:")
    # for text, score in results:
    #     print(f"Score: {score:.4f}")
    #     print(f"Text: {text}\n")


    client = OpenAI(api_key=openai_api_key)
    m = [
        {"role": "developer", "content": f"You will be given a quqery and top 4 retreived documents, you must be a helpful assistant and provide the most relevant useful information to the user. The query will be related to regulation / document retrieval from a set of guidelines designed for BITS Pilani. Do not produce extra information. Your core job is to sythesize the raw data retreived into a coherent and useful response. Aim to be brief in your response. "},
        {"role": "user", "content": f"The query is {userQuery} and the retreived documents are {str(results)}."}
    ]


    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=m
    )

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content
api_key = ""
query("What are the guidelines for the DRC?", api_key)