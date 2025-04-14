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


def group_text(pdf_file):
    delimitered_text = mark_bullet_points_and_table(extract_content_from_pdf(pdf_file))

    # print(bullets)
    split_text = re.split(r'(@BULLET_START|@BULLET_END|@NUMBERED_START|@NUMBERED_END|@TABLE_START|@TABLE_END)', delimitered_text)
    grouped = []
    i = 0
    while i < len(split_text):
        if split_text[i] in ['@BULLET_START', '@NUMBERED_START', '@TABLE_START']:
            content = split_text[i] + split_text[i + 1] + split_text[i + 2]  # start + content + end
            grouped.append(content)
            i += 3
        else:
            grouped.append(split_text[i])
            i += 1
    return grouped

def chunk_text_hybrid(pdf_link: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Chunks text from a PDF with special handling for tables and bullet points.
    Returns a list of dictionaries with keys 'text' and 'source'
    """
    try:
        # Make sure pdf_link is valid and accessible
        split_text = group_text(pdf_link)
        final_chunks = []
        prev_tail = ""
        
        for section in split_text:
            # Handle empty sections
            if not section or not section.strip():
                continue
                
            is_special = (
                section.startswith('@BULLET_START') or
                section.startswith('@NUMBERED_START') or
                section.startswith('@TABLE_START')
            )
            
            if is_special:
                # Add previous tail as context, then full section
                combined = prev_tail + section
                if combined.strip():  # Only add non-empty chunks
                    final_chunks.append({
                        'text': combined,
                        'source': pdf_link
                    })
                
                # Save new tail from this special section
                prev_tail = section[-chunk_overlap:] if len(section) >= chunk_overlap else section
            else:
                # Regular text chunking with overlap
                i = 0
                while i < len(section):
                    chunk = section[i:i + chunk_size]
                    
                    # Prepend overlap only to the first chunk of this section
                    if i == 0:
                        chunk = prev_tail + chunk
                    
                    if chunk.strip():  # Only add non-empty chunks
                        final_chunks.append({
                            'text': chunk,
                            'source': pdf_link
                        })
                    
                    i += chunk_size - chunk_overlap
                
                # Save new tail from last normal chunk
                prev_tail = section[-chunk_overlap:] if len(section) >= chunk_overlap else section
        
        # Debug info
        print(f"Generated {len(final_chunks)} chunks")
        if final_chunks:
            print(f"First chunk type: {type(final_chunks[0])}")
            print(f"First chunk structure: {final_chunks[0]}")
        
        return final_chunks
    except Exception as e:
        print(f"Error in chunk_text_hybrid for {pdf_link}: {str(e)}")
        # Return empty list to avoid crashes
        return []

