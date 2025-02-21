import json
import re
import time
from itertools import product
from collections import defaultdict

# ------------------ Tunable Parameters ------------------
N_GRAM_SIZE = 3  # Change this to adjust the n-gram size globally
MAX_EDIT_DISTANCE = 1  # Change this to adjust the max edit distance globally

# ------------------ Utility Functions ------------------

def load_dictionary(dict_file):
    """Load the dictionary file (one word per line)."""
    with open(dict_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_documents(json_file):
    """Load the JSON documents."""
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------ Soundex Algorithm ------------------

def soundex(word):
    """Compute the Soundex code for a word."""
    word = word.lower()
    mappings = {
        "bfpv": "1", "cgjkqsxz": "2", "dt": "3",
        "l": "4", "mn": "5", "r": "6"
    }
    first_letter = word[0].upper()
    
    encoded = first_letter
    prev_digit = None
    for char in word[1:]:
        for key, value in mappings.items():
            if char in key:
                digit = value
                if digit != prev_digit:
                    encoded += digit
                prev_digit = digit
                break
        else:
            prev_digit = None  
    
    encoded = (encoded + "000")[:4]
    return encoded

def build_soundex_index(dictionary):
    """Create a Soundex dictionary mapping codes to words."""
    soundex_dict = defaultdict(set)
    for word in dictionary:
        code = soundex(word)
        soundex_dict[code].add(word)
    return soundex_dict

# ------------------ N-Gram Similarity ------------------

def get_ngrams(word, n=N_GRAM_SIZE):
    """Return a set of character n-grams for a given word."""
    if len(word) < n:
        return {word}
    return {word[i:i+n] for i in range(len(word)-n+1)}

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def build_dictionary_ngrams(dictionary, n=N_GRAM_SIZE):
    """
    Precompute n-grams for all words in the dictionary.
    Returns a dictionary mapping each word to its set of n-grams.
    """
    dict_ngrams = {}
    for word in dictionary:
        dict_ngrams[word] = get_ngrams(word, n)
    return dict_ngrams

# ------------------ Levenshtein Distance ------------------

def levenshtein_distance(s, t):
    """Compute Levenshtein distance between two strings."""
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s[i-1] == t[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

# ------------------ Combined Spelling Correction ------------------

def correct_word(word, dictionary, soundex_dict, dictionary_ngrams):
    """
    Corrects a single word using Soundex, N-Gram, and Levenshtein Distance.
    """
    soundex_code = soundex(word)
    soundex_candidates = soundex_dict.get(soundex_code, set())

    best_similarity = 0
    best_candidates = set()
    query_ngrams = get_ngrams(word)
    
    for cand in soundex_candidates:
        cand_ngrams = dictionary_ngrams[cand]  
        ngram_sim = jaccard_similarity(query_ngrams, cand_ngrams)
        if ngram_sim > best_similarity:
            best_similarity = ngram_sim
            best_candidates = {cand}
        elif ngram_sim == best_similarity:
            best_candidates.add(cand)

    final_candidates = sorted(best_candidates, key=lambda x: levenshtein_distance(word, x))[:3]
    return final_candidates if final_candidates else [word]

def correct_query(query, dictionary, soundex_dict, dictionary_ngrams):
    """
    Corrects each word in a query and returns top suggestions.
    """
    words = query.split()
    candidate_lists = [correct_word(word, dictionary, soundex_dict, dictionary_ngrams) for word in words]
    return [" ".join(c) for c in product(*candidate_lists)][:10]

# ------------------ Positional Index & Document Retrieval ------------------

def build_positional_index(docs):
    """
    Build a positional index for document fields.
    """
    positional_index = { field: {} for field in ["Title", "Author", "Bibliographic Source", "Abstract"] }
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

def search_documents(query, positional_index):
    """
    Finds documents containing the phrase using the positional index.
    """
    words = query.lower().split()
    if not all(word in positional_index for word in words):
        return []
    return list(set.intersection(*(set(positional_index[word].keys()) for word in words)))

# ------------------ Main Function ------------------

def main():
    dictionary = load_dictionary("dictionary2.txt")
    docs = load_documents("bool_docs.json")
    
    soundex_dict = build_soundex_index(dictionary)
    dictionary_ngrams = build_dictionary_ngrams(dictionary)
    positional_index = build_positional_index(docs)
    
    while True:
        query = input("Enter your query: ").strip().lower()
        if query == "xxx":
            break
        start_time = time.time()
        corrected_queries = correct_query(query, dictionary, soundex_dict, dictionary_ngrams)
        print(corrected_queries)
        print(f"Time taken: {time.time() - start_time:.4f} sec\n")

if __name__ == "__main__":
    main()
