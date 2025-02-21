import json
import time
from combinedLoop import (
    load_dictionary,
    load_documents,
    build_soundex_index,
    build_dictionary_ngrams,
    build_positional_index,
    correct_query
)

TOP_N = 3  # Top N corrections per word


def evaluate_hybrid_spell_checker(test_file, dict_file, docs_file):
    """
    Evaluates the hybrid spelling correction model using Soundex + N-Gram + Levenshtein.
    Measures accuracy, average time per query, and displays top suggestions.
    """
    # Load data
    with open(test_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    dictionary = load_dictionary(dict_file)
    docs = load_documents(docs_file)
    soundex_dict = build_soundex_index(dictionary)
    dictionary_ngrams = build_dictionary_ngrams(dictionary, n=2)
    positional_index = build_positional_index(docs)

    total_queries = len(test_queries)
    correct_count = 0
    total_time = 0.0

    print("\n--- Evaluation Results for Hybrid Spell Checker (Soundex + N-Gram + Levenshtein) ---\n")
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["corrected"]

        start_time = time.time()
        corrections = correct_query(query, dictionary, soundex_dict, dictionary_ngrams)
        elapsed = time.time() - start_time
        total_time += elapsed

        # Check if expected correction is in suggestions
        if expected in corrections:
            correct_count += 1
            print(f"{i}. ✅ Correct: '{query}' → '{expected}' ({elapsed:.4f}s)")
        else:
            print(f"{i}. ❌ Incorrect: '{query}' → Expected: '{expected}', Got: {corrections[:TOP_N]} ({elapsed:.4f}s)")

    # Summary
    accuracy = (correct_count / total_queries) * 100
    avg_time = total_time / total_queries
    print("\n--- Summary ---")
    print(f"Total Queries: {total_queries}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Time per Query: {avg_time:.4f} seconds\n")


if __name__ == "__main__":
    evaluate_hybrid_spell_checker("spell_queries.json", "dictionary2.txt", "bool_docs.json")
