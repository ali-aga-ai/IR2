import json
import time
from Ngram import load_dictionary, load_documents, build_positional_index, ngram_correction

N = 3  # N-gram size

def evaluate_ngram_spell_checker(test_file, dict_file, docs_file):
    # Load data
    with open(test_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    dictionary = load_dictionary(dict_file)
    docs = load_documents(docs_file)
    positional_index = build_positional_index(docs)

    total_queries = len(test_queries)
    correct_count = 0
    total_time = 0.0

    print("\n--- Evaluation Results for N-gram Spell Checker ---\n")
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["corrected"]

        start_time = time.time()
        corrections = ngram_correction(query, dictionary, n=N)
        elapsed = time.time() - start_time
        total_time += elapsed

        # Check if expected correction is in suggestions
        if expected in corrections:
            correct_count += 1
            print(f"{i}. ✅ Correct: '{query}' → '{expected}' ({elapsed:.4f}s)")
        else:
            print(f"{i}. ❌ Incorrect: '{query}' → Expected: '{expected}', Got: {corrections[:3]} ({elapsed:.4f}s)")

    # Summary
    accuracy = (correct_count / total_queries) * 100
    avg_time = total_time / total_queries
    print(f"\n--- Summary ---")
    print(f"Total Queries: {total_queries}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Time per Query: {avg_time:.4f} seconds\n")


if __name__ == "__main__":
    evaluate_ngram_spell_checker("spell_queries.json", "dictionary2.txt", "bool_docs.json")
