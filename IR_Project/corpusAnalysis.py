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

# Extract text from PDF
def extract_content_from_pdf(pdf_path):
    full_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    full_content.append(text)
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
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

# Document Length Distribution
doc_lengths = {doc: len(text.split()) for doc, text in documents.items()}
avg_length = sum(doc_lengths.values()) / len(doc_lengths)

# Shorten document names for readability
def shorten_name(name, length=10):
    return name[:length] + "..." if len(name) > length else name

shortened_names = {shorten_name(doc): length for doc, length in doc_lengths.items()}

# Display document lengths
print("\nDocument Lengths:")
for doc, length in shortened_names.items():
    print(f"{doc}: {length} words")
print(f"\nAverage Document Length: {avg_length:.2f} words\n")

# Plot document lengths
plt.figure(figsize=(8, 5))
sns.barplot(x=list(shortened_names.keys()), y=list(doc_lengths.values()))
plt.title("Document Length Distribution")
plt.ylabel("Number of Words")
plt.xticks(rotation=45, ha="right")
plt.show()

# Most Frequent Terms
all_words = " ".join(documents.values()).split()
word_freq = Counter(all_words)
most_common = word_freq.most_common(10)

# Display top 10 words
print("\nTop 10 Most Frequent Terms:")
for word, freq in most_common:
    print(f"{word}: {freq}")

# Cosine Similarity Matrix
vectorizer = CountVectorizer()
doc_matrix = vectorizer.fit_transform(documents.values())
cos_sim_matrix = cosine_similarity(doc_matrix, doc_matrix)

# Convert to DataFrame with shortened names
cos_sim_df = pd.DataFrame(
    cos_sim_matrix,
    index=[shorten_name(doc) for doc in documents.keys()],
    columns=[shorten_name(doc) for doc in documents.keys()]
)

# Display cosine similarity
print("\nCosine Similarity Between Documents:")
print(cos_sim_df.round(3))

# Heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cos_sim_df, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=True, yticklabels=True)
plt.title("Cosine Similarity Heatmap")
plt.show()
