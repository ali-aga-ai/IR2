import ijson
import json
import math
import time
from memory_profiler import memory_usage
from sortedcontainers import SortedSet

chunk_size = 2000

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
            return chunk_id


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
number_of_chunks = createChunks(chunk_size)
createIndexTable()
end = time.time()
mem_end = memory_usage()[0]

print(f"For chunk size = {chunk_size}")
print(f"Number of chunks created= {number_of_chunks}")
print(f"Time to index = {end - start} seconds")
print(f"Total memory used: {mem_end - mem_start:.2f} MB.")

def bringKObjects(k,index, offset,p): #index is basically konse waale se u wanna bring
    with open(f"index_table{p-1}.json", "r", encoding ="utf-8") as f: 
        items = ijson.items(f,"item")  #item would retreive the 0th object viz a row, item.item retreives output[0][0] 
        for idx, item in enumerate(items):
            if idx < index:
                continue
            result = []
            for i, element in enumerate(item):
                if i < offset - 1:
                    continue
                elif i > offset + k - 1:
                    break
                else:
                    result.append({element: item[element]})  
            
            return result
            

def mergeLists(l1,l2):
    i=0
    j=0
    result = []
    while i < len(l1) and j < len(l2):
            val1 = l1[i]
            val2 = l2[j]            
            if val1 < val2:
                result.append(val1)
                i += 1
            elif val1 > val2:
                result.append(val2)
                j += 1
            else:  # Both are equal
                result.append(val1)
                i += 1
                j += 1
    result.extend(l1[i:])
    result.extend(l2[j:])

    return result  # Return the merged sorted list


def writeToDisk(l, beginning, p):
    with open(f"index_table{p}.json", "a",encoding ="utf-8") as f:
        for idx, item in enumerate(l):
            key, value = next(iter(item.items()))  # Extract key-value pair
            if beginning != 0 or idx > 0:
                f.write(",")  # Add a comma for valid JSON structure
            f.write(f'"{key}": {json.dumps(value)}')  # Write key-value directly


def merge(index1, index2, k, p):
    offset1 = 0
    offset2 = 0
    i, j = 0, 0
    buff1 = None
    buff2 = None
    
    with open(f"index_table{p}.json", "a",encoding ="utf-8") as f:
        f.write("{" if index1 == 0 else "{")

    while True:
        firstEntry = offset1 + offset2
        
        # Only fetch new buffers when needed
        if i == 0:
            buff1 = bringKObjects(k, index1, offset1, p)
        if j == 0:
            buff2 = bringKObjects(k, index2, offset2, p)
            
        # Check termination condition
        if not buff1 and not buff2:
            break
            
        # Handle single buffer cases
        if not buff1:
            writeToDisk(buff2[j:], firstEntry, p)
            break
        if not buff2:
            writeToDisk(buff1[i:], firstEntry, p)
            break
            
        buff3 = []
        # Merge process
        while i < len(buff1) and j < len(buff2):
            key1, val1 = next(iter(buff1[i].items()))
            key2, val2 = next(iter(buff2[j].items()))

            if key1 < key2:
                buff3.append(buff1[i])
                i += 1
            elif key1 > key2:
                buff3.append(buff2[j])
                j += 1
            else:
                merged_value = mergeLists(val1, val2)
                buff3.append({key1: merged_value})
                i += 1
                j += 1

        # Handle buffer exhaustion and reset counters
        if i == len(buff1):
            offset1 += k
            i = 0
        if j == len(buff2):
            offset2 += k
            j = 0
            
        if buff3:
            writeToDisk(buff3, firstEntry, p)

    with open(f"index_table{p}.json", "a",encoding ="utf-8") as f:
        f.write("}")
        

def dumpObject(index,p):
    with open(f"index_table{p-1}.json", "r",encoding ="utf-8") as f:
        items = list(ijson.items(f, "item"))

    if index >= len(items):  #  invalid index
        return  

    with open(f"index_table{p}.json", "a",encoding ="utf-8") as f:
        json.dump([items[index]], f)  



def mergeAll(k, p):
    
    with open(f"index_table{p}.json", "w",encoding ="utf-8") as op:
        op.write("[")
    
    with open(f"index_table{p-1}.json", "r",encoding ="utf-8") as f:
        items = list(ijson.items(f, "item"))
        
    lastIndex = len(items) - 1
    idx = 0
    
    while idx < lastIndex:
        merge(idx, idx + 1, k,p)
        idx += 2
        if idx < lastIndex or (len(items) % 2 != 0):
            with open(f"index_table{p}.json", "a",encoding ="utf-8") as op:
                op.write(",")
    if len(items) % 2 != 0:
        print("hello")
        with open(f"index_table{p}.json", "a",encoding ="utf-8") as f:
            json.dump(items[lastIndex], f)
    
    with open(f"index_table{p}.json", "a",encoding ="utf-8") as op:
        op.write("]")

   


def final(number_of_chunks):
    start_time = time.time()
    mem_start = memory_usage()[0]

    result = math.ceil(math.log2(number_of_chunks))
    for i in range(result + 1):
        mergeAll(5000, i + 1)
# file_path = f"index_table{i-1}.json"
        # if i-1 > 0:
        #     os.remove(file_path)
    end_time = time.time()
    mem_end = memory_usage()[0]

    print(f"Total merging completed in {end_time - start_time:.2f} seconds.")
    print(f"Total memory used: {mem_end - mem_start:.2f} MB.")
        

final(number_of_chunks)
