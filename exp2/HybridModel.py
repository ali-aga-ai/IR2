import json
import re
import time
from itertools import product
from collections import defaultdict

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
    
    # Replace letters with numbers
    encoded = first_letter
    prev_digit = None
    for char in word[1:]:
        for key, value in mappings.items():
            if char in key:
                digit = value
                if digit != prev_digit:  # Avoid consecutive duplicates
                    encoded += digit
                prev_digit = digit
                break
        else:
            prev_digit = None  # Reset for vowels & non-mapped chars
    
    # Pad or truncate to length 4
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

def get_ngrams(word, n=2):
    """Return a set of character n-grams for a given word."""
    if len(word) < n:
        return {word}
    return {word[i:i+n] for i in range(len(word)-n+1)}

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

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

def correct_word(word, dictionary, soundex_dict, max_distance=2, n=2):
    """
    Corrects a single word using Soundex, N-Gram, and Levenshtein Distance.
    First, it finds all candidates sharing the same Soundex code.
    Then, it ranks these candidates by their n-gram Jaccard similarity with the input word.
    Finally, it refines the best matches using Levenshtein Distance and returns up to 3 candidates.
    """
    soundex_code = soundex(word)
    soundex_candidates = soundex_dict.get(soundex_code, set())

    best_similarity = 0
    best_candidates = set()

    for cand in soundex_candidates:
        ngram_sim = jaccard_similarity(get_ngrams(word, n), get_ngrams(cand, n))
        if ngram_sim > best_similarity:
            best_similarity = ngram_sim
            best_candidates = {cand}
        elif ngram_sim == best_similarity:
            best_candidates.add(cand)

    # Refine using Levenshtein Distance
    final_candidates = sorted(best_candidates, key=lambda x: levenshtein_distance(word, x))[:3]
    return final_candidates if final_candidates else [word]

def correct_query(query, dictionary, soundex_dict, max_distance=2, n=2):
    """
    Corrects each word in a query and returns top suggestions.
    The final query suggestions are generated using the Cartesian product of candidate corrections per word.
    The output is limited to the top 10 corrected queries.
    """
    words = query.split()
    candidate_lists = [correct_word(word, dictionary, soundex_dict, max_distance, n) for word in words]
    return [" ".join(c) for c in product(*candidate_lists)][:10]  # Limit to top 10 suggestions

# ------------------ Positional Index & Document Retrieval ------------------

def build_positional_index(docs):
    """Creates a positional index for fast phrase search."""
    index = defaultdict(lambda: defaultdict(list))
    for doc in docs:
        doc_id = doc['Index']
        content = " ".join([doc[key] for key in ["Title", "Author", "Bibliographic Source", "Abstract"] if key in doc]).lower()
        words = re.findall(r'\w+', content)
        for pos, word in enumerate(words):
            index[word][doc_id].append(pos)
    return index

def search_documents(query, positional_index):
    """
    Finds documents containing the phrase using the positional index.
    This function finds documents that contain all the words of the query.
    """
    words = query.lower().split()
    if not all(word in positional_index for word in words):
        return []

    # Find common documents containing all words
    common_docs = set(positional_index[words[0]].keys())
    for word in words[1:]:
        common_docs &= set(positional_index[word].keys())

    # Ensure words appear in order (phrase search)
    matched_docs = []
    for doc in common_docs:
        positions = [positional_index[word][doc] for word in words]
        for start_pos in positions[0]:
            if all(start_pos + i in positions[i] for i in range(1, len(words))):
                matched_docs.append(doc)
                break

    return matched_docs

# ------------------ Display Results ------------------

def show_results(corrected_queries, docs, positional_index):
    """
    Displays corrected queries with matching document indices in the following format:
    'query' found in documents: [doc_id1, doc_id2, ...]
    """
    for corr in corrected_queries:
        matched_docs = search_documents(corr, positional_index)
        indices = [doc["Index"] for doc in docs if doc["Index"] in matched_docs]
        print(f"'{corr}' found in documents: {indices if indices else '(No matches)'}\n")

# ------------------ Main Function ------------------

def main():
    dictionary = load_dictionary("dictionary2.txt")
    docs = load_documents("bool_docs.json")

    # Precompute indices (runs only once)
    soundex_dict = build_soundex_index(dictionary)
    positional_index = build_positional_index(docs)

    print("Hybrid Spelling Correction (Soundex + N-Gram + Levenshtein) with Document Retrieval")
    print("Enter 'xxx' to exit.\n")
    
    while True:
        query = input("Enter your query: ").strip().lower()
        if query == "xxx":
            break
        start_time = time.time()
        corrected_queries = correct_query(query, dictionary, soundex_dict)
        show_results(corrected_queries, docs, positional_index)
        elapsed = time.time() - start_time
        print(f"Time taken for query: {elapsed:.4f} seconds\n")

if __name__ == "__main__":
    main()
