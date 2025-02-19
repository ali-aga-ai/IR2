import ijson
import json

def parse(k):
    chunk_id = 0  # Track chunk number
    i = 0
    with open(r"C:\Documents\code\IR2\Assignment-data\bsbi_docs.json", "r") as f:  
        parser = ijson.items(f, "item") 

        with open("output.json", "w") as out_f:  
            out_f.write("[")  

            first_chunk = True
            while True:
                data = []
                for _ in range(k):  
                    try:
                        data.append(next(parser)) 
                    except StopIteration:
                        break  

                if not data:
                    break  # Stop parsing

                if not first_chunk:
                    out_f.write(",")  # Ensure correct JSON formatting
                json.dump(data, out_f)

                first_chunk = False
                chunk_id += 1  # Increment chunk number
                if(chunk_id == 10):break

            out_f.write("]")  # Close JSON array

# Usage:
parse(5)  # Process in chunks of `k=2`
