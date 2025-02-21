from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import json
import pandas as pd
import pdfplumber
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK resources
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Path to the PDFs folder
pdf_dir = r"pdfs"
documents = {}

def extract_content_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with error handling"""
    full_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text() or ""
                    if text:
                        full_content.append(text)
                except Exception as e:
                    print(f"Warning: Error processing page {page_num}: {str(e)}")
    except Exception as e:
        print(f"Error: Failed to process PDF: {str(e)}")
        return ""
    return "\n".join(full_content)

# Process each PDF
for pdf_file in os.listdir(pdf_dir):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    data = extract_content_from_pdf(pdf_path)
    processed_words = [
        lemmatizer.lemmatize(word, pos="v")
        for word in data.lower().split()
        if word.isalpha() and word not in stop_words
    ]
    documents[pdf_file] = " ".join(processed_words)

# -------------------------------
# 1. Document Length Distribution
# -------------------------------
doc_lengths = {doc: len(text.split()) for doc, text in documents.items()}
avg_length = sum(doc_lengths.values()) / len(doc_lengths)

# Shorten names for table (10 chars) and plot (6 chars)
def shorten_name_table(name, length=10):
    return name[:length] + "..." if len(name) > length else name

def shorten_name_plot(name, length=6):
    return name[:length]

shortened_table_names = {shorten_name_table(doc): length for doc, length in doc_lengths.items()}
shortened_plot_names = [shorten_name_plot(doc) for doc in doc_lengths.keys()]

# Display the table
print("\n📊 Document Lengths:\n")
print(tabulate(shortened_table_names.items(), headers=["Document", "Word Count"], tablefmt="grid"))

# Barplot with very short names for better readability
plt.figure(figsize=(8, 5))
sns.barplot(x=shortened_plot_names, y=list(doc_lengths.values()))
plt.title("Document Length Distribution")
plt.ylabel("Number of Words")
plt.xticks(rotation=45, ha="right")
plt.show()

print(f"\n📌 Average Document Length: {avg_length:.2f} words\n")

# ----------------------
# 2. Most Frequent Terms
# ----------------------
all_words = " ".join(documents.values()).split()
word_freq = Counter(all_words)
most_common = word_freq.most_common(10)

# Display top 10 words in a readable table format
print("\n📊 Top 10 Most Frequent Terms:\n")
print(tabulate(most_common, headers=["Word", "Frequency"], tablefmt="grid"))

# -----------------------------
# 3. Cosine Similarity Matrix
# -----------------------------
# Convert documents into vectors using CountVectorizer for similarity calculation
vectorizer = CountVectorizer()
doc_matrix = vectorizer.fit_transform(documents.values())
cos_sim_matrix = cosine_similarity(doc_matrix, doc_matrix)

# Convert to DataFrame with shortened names
cos_sim_df = pd.DataFrame(
    cos_sim_matrix,
    index=[shorten_name_table(doc) for doc in documents.keys()],
    columns=[shorten_name_table(doc) for doc in documents.keys()]
)

# Display in a clean format
print("\n📌 Cosine Similarity Between Documents:\n")
print(tabulate(cos_sim_df.round(3), headers="keys", tablefmt="fancy_grid"))

# Heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cos_sim_df, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=True, yticklabels=True)
plt.title("Cosine Similarity Heatmap")
plt.show()
