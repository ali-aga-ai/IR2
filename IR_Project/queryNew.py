import faiss
import numpy as np
import openai
import pickle
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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
    
    distances, indices = index.search(query_vector, top_k)
    
    # Add bounds checking
    valid_results = []

    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(pdf_files):
            valid_results.append((pdf_files.iloc[idx]['chunk_text'], distances[0][i], pdf_files.iloc[idx]['pdf_files']))
        else:
            print(f"Warning: Index {idx} is out of bounds")
    return valid_results

def rerank_with_cross_encoder(query, chunk_tuples, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
    """
    chunk_tuples: list of (chunk_text, distance, sourceDoc)
    Returns: list of (chunk_text, distance, sourceDoc, cross_score) sorted by cross_score descending
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    pairs = [(query, chunk_text) for chunk_text, _, _ in chunk_tuples]
    inputs = tokenizer([q for q, c in pairs], [c for q, c in pairs], padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze(-1).cpu().numpy()
    reranked = [
        (chunk_text, distance, sourceDoc, float(score))
        for (chunk_text, distance, sourceDoc), score in zip(chunk_tuples, scores)
    ]
    reranked.sort(key=lambda x: x[3], reverse=True)
    return reranked

def query(userMessages, openai_api_key):
    # Accept both string and list input for userMessages
    if isinstance(userMessages, str):
        userMessages = [{"role": "user", "content": userMessages}]
    client = OpenAI(api_key=openai_api_key)
    userQuery = userMessages[-1]['content']
    systemPrompt = """You’re a retrieval‑query optimizer specialized for a BITS Pilani corpus. 
    Transform any user question into a concise, high-precision query that maximizes finding the exact  section. 
    Follow these steps:
    1. Drop filler (e.g. “please”, “I'd like to know”).
    2. Extract core domain terms
    3. Use official BITS vocabulary (e.g. Regulations, Ordinances, SOP).
    4. Order by specificity (most discriminative terms first).
    5. Output only the final query (4 to 8 words), no extra text.
    6. DO NOT EVER MENTION YOU ARE AN ASSISTANT< YOU ARE SIMPLY A MESSAGE CONVERTER
    7. IN NO CIRCUMSTANCE MUST YOU RESPOND WITH AN EMPTY STRING

    Examples:
    Q: “Could you tell me the late fee policy for library books at BITS?”
    A: “Library fine policy regulations”

    Q: “What’s the procedure to apply for summer internship credits?”
    A: “Summer internship credit SOP”
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
    print("Updated Query", updatedQuery)

    results = query_faiss(updatedQuery, openai_api_key)

    # Rerank using Cross Encoder
    reranked = rerank_with_cross_encoder(updatedQuery, results)

    resultString = ""
    for i, (chunk_text, distance, sourceDoc, cross_score) in enumerate(reranked):
        resultString += (
            f'{i}th Retreived chunk:{chunk_text}... its cosine distance from query vector {distance} its source document {sourceDoc}\n'
        )

    m = [
        {"role": "system", "content": f"""You are given:
            - A user query about BITS Pilani regulations.
            - The top-k retrieved text segments, each tagged with its source filename.

            Your task:
            - Synthesize only the relevant information from those segments.
           
            Formatting (exactly):
            Answer: <your brief synthesis here>

            Citation:
            For each fact, append “SOURCE: <filename>” Each Source MUST BE on a new line.
         
            In source have only the source and nothing more. in the answer there must be no mention of the source.
            """},
        {"role": "user", "content": f"The query is {userQuery} and the retreived documents are {resultString}."}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=m
    )
    response = (completion.choices[0].message.content)
    return response

