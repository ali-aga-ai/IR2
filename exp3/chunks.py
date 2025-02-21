import time
import ijson
import json
from sortedcontainers import SortedSet
import psutil 
import os
from memory_profiler import memory_usage

chunk_size = 10000

def createChunks(k):
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

            out_f.write("]")  # Close JSON array
            print(chunk_id)


def createIndexTable():
    with open("output.json", "r") as f, open("index_table0.json", "w") as out_f: 

        for i, item in enumerate(ijson.items(f, "item")):
            index_table = {}
            if i == 0 : out_f.write("[")
            else: out_f.write(",")
            
            for k in range(len(item)): # len(item) instead of chunk size ensures no out of bounds

                for word in item[k]["Abstract"].split():
                    word = word.replace('\\', '').replace('"', '').strip()  # Remove backslashes and double quotes
                    
                    if word not in index_table:
                        index_table[word] = SortedSet([int(item[k]["Index"])])  # Convert to int
                    else:
                        index_table[word].add(int(item[k]["Index"]))  # Ensure int before adding

                if i + 1 >= chunk_size:  # Stop after processing `chunk_size` elements
                    break
            sorted_index_table = {key: list(value) for key, value in sorted(index_table.items())}
            json.dump(sorted_index_table, out_f)   
        
        out_f.write("]")
        # Convert to JSON-friendly format **only when dumping JSON**

start = time.time()
mem_start = memory_usage()[0]
createChunks(chunk_size)
createIndexTable()
end = time.time()
mem_end = memory_usage()[0]

print(f"For chunk size = {chunk_size}")
print(f"Time to index = {start - end} seconds")
print(f"Total memory used: {mem_end - mem_start:.2f} MB.")
