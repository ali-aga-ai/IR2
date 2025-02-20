import ijson
import json
from sortedcontainers import SortedSet

chunk_size = 5000

with open("output.json", "r") as f, open("index_table0.json", "w") as out_f: 

    for i, item in enumerate(ijson.items(f, "item")):
        index_table = {}
        if i == 0 : out_f.write("[")
        else: out_f.write(",")
        
        for k in range(len(item)):  # Safely iterate only over available items in the chunk

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
