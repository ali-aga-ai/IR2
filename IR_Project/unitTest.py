import re
import pdfplumber
import pandas as pd

def extract_content_from_pdf(pdf_path: str) -> str: # here... could extract bullet points separately from text and tables, and append separately to table
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
                        df = pd.DataFrame(table).fillna('').replace(r'^\s*$', '', regex=True) # remove empty strings 
                        full_content.append(f"\n@TABLE_START:\n{df.to_string(index=False, header=False)}\n@TABLE_END\n") # this will convert the table to a string and remove the index and header. The index is the row number, and the header is the column names. This is done to make the table easier to read.
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""
    return "\n".join(full_content) # this will join all the text and tables extracted from the pdf into a single string. The tables are formatted as strings with headers and indices removed. 


def mark_bullet_points_and_table(text):
    # Find bullet points groups
    bullet_pattern = r'(?:^|\n)(?:[ \t]*(?:•|\-|\*)[ \t].*(?:\n[ \t]+.*)*\n?)+'
    
    # Replace with delimiters
    text = re.sub(bullet_pattern, r'\n@BULLET_START\n\g<0>\n@BULLET_END\n', text)
    
    # Find numbered list groups
    numbered_pattern = r'(?:^|\n)(?:[ \t]*\d+\.[ \t].*(?:\n[ \t]+.*)*\n?)+'
    
    # Replace with delimiters
    text = re.sub(numbered_pattern, r'\n@NUMBERED_START\n\g<0>\n@NUMBERED_END\n', text)
    
    return text


delimitered_text = mark_bullet_points_and_table(extract_content_from_pdf("C:\Documents\code\IR2\IR_Project\pdfs\DRC_Guidelines-2015-updated.pdf"))

# print(bullets)
split_text = re.split(r'(@BULLET_START|@BULLET_END|@NUMBERED_START|@NUMBERED_END|@TABLE_START|@TABLE_END)', delimitered_text)
print(split_text)
grouped = []
i = 0
while i < len(split_text):
    if split_text[i] in ['@BULLET_START', '@NUMBERED_START', '@TABLE_START']:
        start_tag = split_text[i]
        content = split_text[i] + split_text[i + 1] + split_text[i + 2]  # start + content + end
        grouped.append(content)
        i += 3
    else:
        grouped.append(split_text[i])
        i += 1

def chunk_text(split_text, chunk_size=500, chunk_overlap=50):
    final_chunks = []
    for section in split_text:
        if section.startswith('@BULLET_START') or section.startswith('@NUMBERED_START') or section.startswith('@TABLE_START'):
            final_chunks.append(section)
        else:
            i = 0
            while i < len(section):
                final_chunks.append(section[i:i + chunk_size])
                i += chunk_size - chunk_overlap  # step forward with overlap
    return final_chunks

print(chunk_text(grouped))