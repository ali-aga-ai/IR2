from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import json
from sortedcontainers import SortedSet
import time
from memory_profiler import memory_usage

# nltk.download("wordnet")
# nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

OPERATORS = {'AND', 'OR', 'NOT'}

def preprocesing(input_data_path, output_index_path):
    start_time = time.time()
    mem_start = memory_usage()[0]

    inverted_index = {}

    with open(input_data_path, "r") as f:
        data = json.load(f)

    i = 1

    for obj in data:
        words = obj["Abstract"].lower().split()  
        processed_words = [
            lemmatizer.lemmatize(word, pos="v")  
            for word in words if word not in stop_words  
        ]
        for word in processed_words:
            if word not in inverted_index:
                inverted_index[word] = SortedSet([i])
            else:
                inverted_index[word].add(i)
        obj["Abstract"] = " ".join(processed_words)  
        i += 1
        
    for key, value in inverted_index.items():
        inverted_index[key] = list(value)

    with open(output_index_path, "w") as f:
        f.write(json.dumps(inverted_index, indent=1))
    
    end_time = time.time()
    mem_end = memory_usage()[0]

    print(f"time taken for indexing: {end_time - start_time:.2f} seconds")
    print(f"memory used for preprocessing: {mem_end - mem_start:.2f} MB")


def preprocess_term(term):
    term = term.lower()
    if term not in stop_words:
        return lemmatizer.lemmatize(term, pos="v")
    return ""


def evaluate_query(query, inverted_index, all_docs):
    query = query.strip() #removing leading and trailing whitespaces (not really needed)
    if query.startswith('(') and query.endswith(')'):
        return evaluate_query(query[1:-1], inverted_index, all_docs)

    if ' AND ' in query:
        left, right = query.split(' AND ', 1)  #splits the string at the first occurence of AND
        return evaluate_query(left, inverted_index, all_docs) & evaluate_query(right, inverted_index, all_docs)
    elif ' OR ' in query:
        left, right = query.split(' OR ', 1)
        return evaluate_query(left, inverted_index, all_docs) | evaluate_query(right, inverted_index, all_docs)
    elif query.startswith('NOT '):
        return all_docs - evaluate_query(query[4:], inverted_index, all_docs)
    else:
        preprocessed_term = preprocess_term(query)
        return set(inverted_index.get(preprocessed_term, [])) if preprocessed_term else set()


def query(data_path, index_path, query_path, results_path):
    start_time = time.time()
    mem_start = memory_usage()[0]

    with open(data_path, "r") as f:
        data = json.load(f)

    all_docs = set(range(1, len(data) + 1))

    with open(index_path, "r") as f:
        inverted_index = json.load(f)

    with open(query_path, "r") as f:
        ans = []
        for line in f: 
            query = line.strip()
            result = evaluate_query(query, inverted_index, all_docs)
            ans.append(result)
        with open(results_path, "w") as f:
            for i, answer in enumerate(ans):
                f.write(f"for query {i} the documents are {answer}\n")
    
    end_time = time.time()
    mem_end = memory_usage()[0]
    
    print(f"Time taken for querying: {end_time - start_time:.2f} seconds")
    print(f"Memory used for querying: {mem_end - mem_start:.2f} MB")


input_data_path = input("Enter the path for input data (bool_docs.json): ")
output_index_path = input("Enter the path to save the inverted index (output.json): ")
query_path = input("Enter the path for query file (bool_queries.txt): ")
results_path = input("Enter the path to save the results (results.txt): ")

preprocesing(input_data_path, output_index_path)
query(input_data_path, output_index_path, query_path, results_path)

with open("output.json", "r") as f:
    size = len(json.load(f))
    print(size)