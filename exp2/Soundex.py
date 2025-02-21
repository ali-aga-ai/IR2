import json
import re
import time
from itertools import product

# --- Configuration ---
TOP_N = 3  # Limit the number of candidate corrections per word
MAX_OUTPUT_QUERIES = 10  # Limit the number of corrected queries displayed

# --- Common helper functions ---

def load_dictionary(dict_file):
    """Load the dictionary file (one word per line)."""
    with open(dict_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_documents(json_file):
    """Load the JSON documents."""
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def build_positional_index(docs):
    """
    Build a positional index for each field (Title, Author, Bibliographic Source, Abstract).
    Stores a mapping from word → { doc_id → set(positions) }.
    """
    positional_index = { field: {} for field in ["Title", "Author", "Bibliographic Source", "Abstract"] }
    for doc in docs:
        doc_id = doc["Index"]
        for field in ["Title", "Author", "Bibliographic Source", "Abstract"]:
            if field in doc:
                words = re.findall(r'\b[a-zA-Z]+\b', doc[field].lower())
                for pos, word in enumerate(words):
                    if word not in positional_index[field]:
                        positional_index[field][word] = {}
                    if doc_id not in positional_index[field][word]:
                        positional_index[field][word][doc_id] = set()
                    positional_index[field][word][doc_id].add(pos)
    return positional_index

def get_frequency(word, positional_index):
    """
    Compute the frequency of a word across all fields in the positional index.
    Frequency = Total number of positions where the word appears.
    """
    frequency = 0
    for field_index in positional_index.values():
        if word in field_index:
            for positions in field_index[word].values():
                frequency += len(positions)
    return frequency

def phrase_in_doc(field_index, doc_id, phrase_words):
    """
    Check if all words in phrase_words appear in the document (order doesn't matter).
    """
    return all(word in field_index and doc_id in field_index[word] for word in phrase_words)

def find_docs_with_phrase_positional(docs, positional_index, phrase):
    """
    Using positional indexing, return a list of documents where all words in the phrase appear.
    Order of words does NOT matter.
    """
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

def show_results(corrections, docs, positional_index):
    """Display up to MAX_OUTPUT_QUERIES corrected phrases and document indices."""
    print("\nPossible corrections (showing first 10 only):\n")
    for corr in corrections[:MAX_OUTPUT_QUERIES]:
        matched_docs = find_docs_with_phrase_positional(docs, positional_index, corr)
        indices = [doc["Index"] for doc in matched_docs]
        print(f"'{corr}' found in documents: {indices if indices else '(No matches)'}\n")

# --- Soundex Spell Checker Implementation ---

def soundex(word):
    """Compute the Soundex code for a given word."""
    word = word.upper()
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }
    code = word[0]
    prev = mapping.get(word[0], "")
    for char in word[1:]:
        curr = mapping.get(char, "0")
        if curr != "0" and curr != prev:
            code += curr
        prev = curr
    return code.ljust(4, "0")[:4]

def soundex_correction(query, dictionary, positional_index):
    """
    Generate corrected phrases based on Soundex codes, selecting top N most frequent words.
    """
    soundex_map = {}
    for word in dictionary:
        code = soundex(word)
        soundex_map.setdefault(code, []).append(word)

    words = query.split()
    candidate_lists = []
    for w in words:
        candidates = soundex_map.get(soundex(w), [w])  # Get candidates
        # Rank by frequency and select top N
        candidates = sorted(candidates, key=lambda word: get_frequency(word, positional_index), reverse=True)[:TOP_N]
        candidate_lists.append(candidates)

    # Generate corrected phrases (Cartesian product)
    corrected_phrases = [" ".join(c) for c in product(*candidate_lists)]
    return corrected_phrases

# --- Main function for Soundex ---

def main():
    dictionary = load_dictionary("dictionary2.txt")
    docs = load_documents("bool_docs.json")
    # Preprocessing: Build the positional index (done once)
    positional_index = build_positional_index(docs)
    
    print("Soundex-based Spell Checker. Type 'xxx' to exit.\n")
    while True:
        query = input("Enter your query: ").strip().lower()
        if query == "xxx":
            break
        start_time = time.time()
        corrections = soundex_correction(query, dictionary, positional_index)
        show_results(corrections, docs, positional_index)
        elapsed = time.time() - start_time
        print(f"Time taken for query: {elapsed:.4f} seconds\n")
        
if __name__ == "__main__":
    main()
