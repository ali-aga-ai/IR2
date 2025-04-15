import os
import re
import json
import nltk
import pdfplumber
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sortedcontainers import SortedSet

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

### ------------------------ Hybrid Chunking Functions ------------------------

def extract_content_from_pdf(pdf_path: str) -> str:
    full_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    full_content.append(text)
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table).fillna('').replace(r'^\s*$', '', regex=True)
                        full_content.append(f"\n@TABLE_START:\n{df.to_string(index=False, header=False)}\n@TABLE_END\n")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""
    return "\n".join(full_content)

def mark_bullet_points_and_table(text):
    bullet_pattern = r'(?:^|\n)(?:[ \t]*(?:•|\-|\*)[ \t].*(?:\n[ \t]+.*)*\n?)+'
    text = re.sub(bullet_pattern, r'\n@BULLET_START\n\g<0>\n@BULLET_END\n', text)
    numbered_pattern = r'(?:^|\n)(?:[ \t]*\d+\.[ \t].*(?:\n[ \t]+.*)*\n?)+'
    text = re.sub(numbered_pattern, r'\n@NUMBERED_START\n\g<0>\n@NUMBERED_END\n', text)
    return text

def group_text(pdf_file):
    delimitered_text = mark_bullet_points_and_table(extract_content_from_pdf(pdf_file))
    split_text = re.split(r'(@BULLET_START|@BULLET_END|@NUMBERED_START|@NUMBERED_END|@TABLE_START|@TABLE_END)', delimitered_text)
    grouped = []
    i = 0
    while i < len(split_text):
        if split_text[i] in ['@BULLET_START', '@NUMBERED_START', '@TABLE_START']:
            content = split_text[i] + split_text[i + 1] + split_text[i + 2]
            grouped.append(content)
            i += 3
        else:
            grouped.append(split_text[i])
            i += 1
    return grouped

def chunk_text_hybrid(pdf_path: str, chunk_size=1500, chunk_overlap=500) -> list[dict]:
    split_text = group_text(pdf_path)
    final_chunks = []
    prev_tail = ""

    for section in split_text:
        if not section or not section.strip():
            continue
        is_special = section.startswith('@BULLET_START') or section.startswith('@NUMBERED_START') or section.startswith('@TABLE_START')
        if is_special:
            combined = prev_tail + section
            final_chunks.append({'text': combined, 'source': pdf_path})
            prev_tail = section[-chunk_overlap:] if len(section) >= chunk_overlap else section
        else:
            i = 0
            while i < len(section):
                chunk = section[i:i + chunk_size]
                if i == 0:
                    chunk = prev_tail + chunk
                final_chunks.append({'text': chunk, 'source': pdf_path})
                i += chunk_size - chunk_overlap
            prev_tail = section[-chunk_overlap:] if len(section) >= chunk_overlap else section

    return final_chunks

### ------------------------ Boolean Indexing ------------------------

def tokenize(text):
    words = text.lower().split()
    return [
        lemmatizer.lemmatize(word, pos="v")
        for word in words if word.isalpha() and word not in stop_words
    ]

def build_boolean_index(pdf_folder_path: str, chunk_size=1500, chunk_overlap=500):
    inverted_index = {}
    chunk_id_table = {}  # Map of chunk_id -> chunk content

    chunk_counter = 0
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)  # ✅ Full path to each PDF
            chunks = chunk_text_hybrid(file_path, chunk_size, chunk_overlap)

            for chunk in chunks:
                chunk_id = f"{filename}::chunk_{chunk_counter}"
                chunk_id_table[chunk_id] = chunk["text"]
                tokens = tokenize(chunk["text"])

                for token in tokens:
                    if token not in inverted_index:
                        inverted_index[token] = SortedSet([chunk_id])
                    else:
                        inverted_index[token].add(chunk_id)

                chunk_counter += 1

    # Save to JSON in current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "chunk_boolean_index.json")
    chunk_data_path = os.path.join(current_dir, "chunk_id_table.json")

    with open(index_path, "w") as f:
        json.dump({k: list(v) for k, v in inverted_index.items()}, f, indent=2)

    with open(chunk_data_path, "w") as f:
        json.dump(chunk_id_table, f, indent=2)

    print(f"✅ Boolean index created with {len(inverted_index)} unique terms and {chunk_counter} chunks.")
    print(f"✅ Index files saved to {current_dir}")
    return inverted_index, chunk_id_table


if __name__ == "__main__":
    # Automatically resolve path to ../pdfs from inside /boolean_model/
    pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdfs"))
    build_boolean_index(pdf_dir)
