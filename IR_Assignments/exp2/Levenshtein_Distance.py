import json
import re
import time
from itertools import product

# Maximum edit distance for corrections
K = 5

# Load dictionary and documents
def load_dictionary(dict_file):
    with open(dict_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_documents(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

# Build positional index for document fields
def build_positional_index(docs):
    positional_index = {field: {} for field in ["Title", "Author", "Bibliographic Source", "Abstract"]}
    for doc in docs:
        doc_id = doc["Index"]
        for field in positional_index:
            if field in doc:
                words = re.findall(r'\b[a-zA-Z]+\b', doc[field].lower())
                for pos, word in enumerate(words):
                    if word not in positional_index[field]:
                        positional_index[field][word] = {}
                    if doc_id not in positional_index[field][word]:
                        positional_index[field][word][doc_id] = set()
                    positional_index[field][word][doc_id].add(pos)
    return positional_index

# Check if all words in a phrase appear in a document
def phrase_in_doc(field_index, doc_id, phrase_words):
    return all(word in field_index and doc_id in field_index[word] for word in phrase_words)

# Find documents containing all words in a phrase
def find_docs_with_phrase_positional(docs, positional_index, phrase):
    matched_docs = []
    phrase_words = re.findall(r'\b[a-zA-Z]+\b', phrase.lower())
    for doc in docs:
        doc_id = doc["Index"]
        found_in_field = False
        for field in ["Title", "Author", "Bibliographic Source", "Abstract"]:
            field_index = positional_index.get(field, {})
            if phrase_in_doc(field_index, doc_id, phrase_words):
                found_in_field = True
                break
        if found_in_field:
            matched_docs.append(doc)
    return matched_docs

# Display corrected queries and matching documents
def show_results(corrections, docs, positional_index):
    print("\nPossible corrections (showing first 10 only):\n")
    for corr in corrections[:10]:
        matched_docs = find_docs_with_phrase_positional(docs, positional_index, corr)
        indices = [doc["Index"] for doc in matched_docs]
        print(f"'{corr}' found in documents: {indices if indices else '(No matches)'}\n")

# Levenshtein distance calculation
def levenshtein_distance(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

# Edit distance-based correction
def edit_distance_correction(query, dictionary, max_distance=2):
    words = query.split()
    candidate_lists = []
    for w in words:
        min_dist = float('inf')
        candidates = []
        for cand in dictionary:
            d = levenshtein_distance(w, cand)
            if d < min_dist:
                min_dist = d
                candidates = [cand]
            elif d == min_dist:
                candidates.append(cand)
        if min_dist <= max_distance:
            candidate_lists.append(candidates)
        else:
            candidate_lists.append([w])
    corrected_phrases = [" ".join(c) for c in product(*candidate_lists)]
    return corrected_phrases

# Main function
def main():
    dictionary = load_dictionary("dictionary2.txt")
    docs = load_documents("bool_docs.json")
    positional_index = build_positional_index(docs)

    print("Edit Distance-based Spell Checker. Type 'xxx' to exit.\n")
    while True:
        query = input("Enter your query: ").strip().lower()
        if query == "xxx":
            break
        start_time = time.time()
        corrections = edit_distance_correction(query, dictionary, max_distance=K)
        show_results(corrections, docs, positional_index)
        elapsed = time.time() - start_time
        print(f"Time taken for query: {elapsed:.4f} seconds\n")

if __name__ == "__main__":
    main()
