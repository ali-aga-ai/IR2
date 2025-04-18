import faiss
import numpy as np
import openai
import pickle
from openai import OpenAI


def get_embedding(text, api_key,model="text-embedding-ada-002"):

    client = OpenAI(api_key=api_key)
    text = text.replace("\n", " ")

    return client.embeddings.create(input = [text], model=model).data[0].embedding

def query_faiss(query_text, api_key, top_k=7):

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
            valid_results.append((pdf_files.iloc[idx]['chunk_text'], distances[0][i], pdf_files.iloc[idx]['pdf_files']))
            # print(pdf_files.iloc[idx]['pdf_files'])
        else:
            print(f"Warning: Index {idx} is out of bounds")
    # print(str(valid_results))
    return valid_results


def query(userMessages, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    userQuery = userMessages[-1]['content']
    print("User Query",userQuery)
    systemPrompt = """You’re a retrieval‑query optimizer specialized for a BITS Pilani guidelines corpus. 
    Transform any user question into a concise, high-precision query that maximizes finding the exact guideline section. 
    Follow these steps:
    1. Drop filler (e.g. “please”, “I'd like to know”).
    2. Extract core domain terms
    3. Use official BITS vocabulary (e.g. Regulations, Ordinances, SOP).
    4. Order by specificity (most discriminative terms first).
    5. Output only the final query (4 to 8 words), no extra text.

    Examples:
    Q: “Could you tell me the late fee policy for library books at BITS?”
    A: “BITS Pilani library fine policy regulations”

    Q: “What’s the procedure to apply for summer internship credits?”
    A: “BITS Pilani summer internship credit SOP”
    """

    messages = [{"role": "system", "content": systemPrompt}] + userMessages
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages
        )

    except Exception as e:
        print("Error during OpenAI API call:", e)
        return "Failed to generate query."


    updatedQuery = response.choices[0].message.content
    print("Updated Query" , updatedQuery)

    results = query_faiss(updatedQuery,openai_api_key)
    # print("\nResults:")
    # for text, score in results:
    #     print(f"Score: {score:.4f}")
    #     print(f"Text: {text}\n")

    resultString = ""
    for i, (chunk_text, distance, sourceDoc) in enumerate(results): 
        resultString+=(f'{i}th Retreived chunk:{chunk_text}... its cosine distance from query vector {distance} its source document {sourceDoc}\n')
    # print(resultString)

    m = [
        {"role": "system", "content": f"You will be given a query and top k retreived segments alongside their file location, you must be a helpful assistant and provide the most relevant useful information to the user. The query will be related to regulation / document retrieval from a set of guidelines designed for BITS Pilani. Do not produce extra information. Try to be brief in responses. Your core job is to sythesize the raw data retreived into a coherent and useful response. For every segment used cite its source, in the format SOURCE: filename."},
        {"role": "user", "content": f"The query is {userQuery} and the retreived documents are {resultString}."}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=m
    )
    print("Model Response: ")
    response = (completion.choices[0].message.content)
    userMessages.append({"role": "assistant", "content": response})
    return userMessages
# api_key = ""
# query([{"role": "user", "content" : "For submitting an industry research proposal, what are the different budget heads?"}], api_key)



# ''' a very rough architecture for caching queries for better retreival (very bad for now)'''

# cached_queries = { 
#     "query1_embedding" : {"clicked document 1 ": "numTimes clcked", 
#                 "clicked document 2": "num times clicked"},
# }

# def buildIndexOfcachedQueries():
#     # Get the first query's embedding to infer the dimension
#     first_query_embedding = list(cached_queries.keys())[0]
#     embedding_dim = first_query_embedding.shape[0]

#     # Create FAISS index for cosine similarity
#     index = faiss.IndexFlatIP(embedding_dim)  # Index for inner product (cosine similarity)

#     # Loop through cached queries
#     for query_embedding, clicked_documents in cached_queries.items():
        
#         # Add query's embedding to FAISS index
#         index.add(query_embedding.reshape(1, -1))

#     print(f"Index has {index.ntotal} vectors.")
#     return index


# def improve_score(user_query_embeddding, retreivedSegments):

#     index = buildIndexOfcachedQueries() 
    
#     threshold = 0.5
#     distances, indices = index.search(user_query_embeddding, top_k =5)
#     #we would know the source doc id of thre retreived segmenrs
#     valid_indices = [i for i, dist in zip(indices[0], distances[0]) if dist <= threshold]

#     boosts = {}
#     total_boost = 0

#     #asssigning a boost score to each clicked document for similiar queries (this is a very poor idea)
#     for i, query in enumerate(cached_queries.items()):
#         if i in valid_indices:
#             for document_id,times_clicked in query.items():
#                 boosts[document_id]+=times_clicked
#                 total_boost+=times_clicked

#     for key,value in boosts.items():
#         boosts[key] = (1+ value/total_boost) 

#     for segment in retreivedSegments:
#         source_doc = segment["source"]
#         if source_doc in set(boosts.keys()):
#             segment["distance"] *= boosts[source_doc]

