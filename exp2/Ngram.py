import json
import re
import time
from itertools import product

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

# N-gram similarity functions
def get_ngrams(word, n=2):
    if len(word) < n:
        return {word}
    return {word[i:i+n] for i in range(len(word)-n+1)}

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

# N-gram based correction
def ngram_correction(query, dictionary, n=2):
    words = query.split()
    candidate_lists = []
    for w in words:
        query_ngrams = get_ngrams(w, n)
        best_similarity = 0
        candidates = []
        for cand in dictionary:
            cand_ngrams = get_ngrams(cand, n)
            similarity = jaccard_similarity(query_ngrams, cand_ngrams)
            if similarity > best_similarity:
                best_similarity = similarity
                candidates = [cand]
            elif similarity == best_similarity:
                candidates.append(cand)
        candidate_lists.append(candidates if candidates else [w])
    corrected_phrases = [" ".join(c) for c in product(*candidate_lists)]
    return corrected_phrases

# Main function
def main():
    dictionary = load_dictionary("dictionary2.txt")
    docs = load_documents("bool_docs.json")
    positional_index = build_positional_index(docs)

    print("N-gram based Spell Checker. Type 'xxx' to exit.\n")
    while True:
        query = input("Enter your query: ").strip().lower()
        if query == "xxx":
            break
        start_time = time.time()
        corrections = ngram_correction(query, dictionary, n=2)
        show_results(corrections, docs, positional_index)
        elapsed = time.time() - start_time
        print(f"Time taken for query: {elapsed:.4f} seconds\n")

if __name__ == "__main__":
    main()
