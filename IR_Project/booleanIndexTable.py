from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import json
import pandas as pd
import pdfplumber
from sortedcontainers import SortedSet
import os

# hash table for docid
doc_id_table = {
    "BITS-Pilani-International-Travel-Award_Guidelines-1.pdf": 1,
    "CheckList_PhD-Thesis-submission.pdf": 2,
    "Documents_required.pdf": 3,
    "DRC_Guidelines-2015-updated.pdf": 4,
    "Guidelines_for-PhD-Proposal.pdf": 5,
    "Leave-policy-for-the-institute-supported-PhD-students.pdf": 6,
}

inverted_index = {}

nltk.download("wordnet")
nltk.download("stopwords")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def extract_content_from_pdf(pdf_path: str) -> str:
    """Extract text and tables from PDF with improved error handling"""
    full_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text
                    text = page.extract_text() or ""
                    if text:
                        full_content.append(text)
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            df = pd.DataFrame(table).fillna('').replace(r'^\s*$', '', regex=True)
                            full_content.append(f"\nTable Content:\n{df.to_string(index=False, header=False)}\n")
                except Exception as e:
                    print(f"Warning: Error processing page {page_num} in {pdf_path}: {str(e)}")
    except Exception as e:
        print(f"Error: Failed to process PDF {pdf_path}: {str(e)}")
        return ""
    
    return "\n".join(full_content)

# Path to the PDFs folder
pdf_dir = r"C:\Documents\code\IR2\IR_Project\pdfs"

# Process each PDF
for pdf_file in os.listdir(pdf_dir):
    pdf_path = os.path.join(pdf_dir, pdf_file)
    
    if pdf_file not in doc_id_table:
        print(f"Skipping unrecognized file: {pdf_file}")
        continue
    
    data = extract_content_from_pdf(pdf_path)
    
    processed_words = [
        lemmatizer.lemmatize(word, pos="v")  
        for word in data.lower().split()
        if word.isalpha() and word not in stop_words  
    ]
    
    for word in processed_words:
        if word not in inverted_index:
            inverted_index[word] = SortedSet([doc_id_table[pdf_file]])
        else:
            inverted_index[word].add(doc_id_table[pdf_file])

#sorted set is not "json serializable"
inverted_index = {key: list(value) for key, value in inverted_index.items()}

with open("output.json", "w") as f:
    json.dump(inverted_index, f, indent=1)

with open("output.json", "r") as f:
    inverted_index = json.load(f)

vocab_size = len(inverted_index)
print(f"vocab size: {vocab_size}")
