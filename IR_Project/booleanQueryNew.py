import json
import os
from sortedcontainers import SortedSet

# Load the Boolean index and chunk_id_table from the JSON files
def load_index_and_chunk_table(current_dir):
    index_path = os.path.join(current_dir, "chunk_boolean_index.json")
    chunk_data_path = os.path.join(current_dir, "chunk_id_table.json")

    if not os.path.exists(index_path) or not os.path.exists(chunk_data_path):
        print("Index or chunk data not found! Please build the index first.")
        return None, None

    with open(index_path, 'r') as f:
        inverted_index = json.load(f)

    with open(chunk_data_path, 'r') as f:
        chunk_id_table = json.load(f)

    return inverted_index, chunk_id_table

def tokenize_query(query):
    tokens = []
    current_token = ""
    # Use symbols for operators
    operator_map = {'&&': 'AND', '||': 'OR', '~': 'NOT', '(': '(', ')': ')'}
    symbols = set(operator_map.keys()).union({'(', ')'})
    i = 0
    query = query.strip()
    while i < len(query):
        if query[i].isspace():
            if current_token:
                tokens.append(current_token.lower())
                current_token = ""
            i += 1
            continue
        # Check for multi-char operators first
        if query[i:i+2] in operator_map:
            if current_token:
                tokens.append(current_token.lower())
                current_token = ""
            tokens.append(operator_map[query[i:i+2]])
            i += 2
            continue
        elif query[i] in operator_map:
            if current_token:
                tokens.append(current_token.lower())
                current_token = ""
            tokens.append(operator_map[query[i]])
            i += 1
            continue
        else:
            current_token += query[i]
            i += 1
    if current_token:
        tokens.append(current_token.lower())
    return tokens

def infix_to_postfix(tokens):
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0}
    output = []
    operator_stack = []
    
    for token in tokens:
        if token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()
        elif token in {'AND', 'OR', 'NOT'}:
            while (operator_stack and operator_stack[-1] != '(' and 
                   precedence[operator_stack[-1]] >= precedence[token]):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        else:
            output.append(token)
    
    while operator_stack:
        output.append(operator_stack.pop())
    
    return output

def evaluate_postfix(postfix, inverted_index, total_chunks):
    stack = []
    all_docs = set(range(len(total_chunks)))
    
    for token in postfix:
        if token == 'AND':
            op2 = stack.pop()
            op1 = stack.pop()
            stack.append(op1.intersection(op2))
        elif token == 'OR':
            op2 = stack.pop()
            op1 = stack.pop()
            stack.append(op1.union(op2))
        elif token == 'NOT':
            op1 = stack.pop()
            stack.append(all_docs - op1)
        else:
            if token in inverted_index:
                stack.append(set(inverted_index[token]))
            else:
                stack.append(set())
    
    return stack[0] if stack else set()

# Function to query the Boolean index
def query_boolean_index(inverted_index, chunk_id_table, query):
    tokens = tokenize_query(query)
    postfix = infix_to_postfix(tokens)
    result_chunks = evaluate_postfix(postfix, inverted_index, chunk_id_table)
    
    results = []
    for chunk_id in result_chunks:
        chunk_text = chunk_id_table.get(str(chunk_id), "")  # Convert chunk_id to string
        results.append({"chunk_id": chunk_id, "text": chunk_text[:500] + "..."})
    
    return results

# Simple user interface for querying
def run_query_interface(pdf_folder_path):
    inverted_index, chunk_id_table = load_index_and_chunk_table(pdf_folder_path)

    if not inverted_index or not chunk_id_table:
        return

    print("Welcome to the Query Interface!")
    print("Type 'exit' to quit the interface.")
    
    while True:
        query = input("\nEnter query: ")
        if query.lower() == "exit":
            break

        results = query_boolean_index(inverted_index, chunk_id_table, query)
        
        if results:
            print(f"\nFound {len(results)} matching chunks:")
            for result in results:
                print(f"Chunk ID: {result['chunk_id']}\nSnippet: {result['text']}")
        else:
            print("No results found for your query.")

if __name__ == "__main__":
    # Use current directory for index files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_query_interface(script_dir)
