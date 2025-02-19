import ijson
import json
import os
import math

def bringKObjects(k,index, offset,p): #index is basically konse waale se u wanna bring
    with open(f"index_table{p-1}.json", "r") as f: 
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
    with open(f"index_table{p}.json", "a") as f:
        for idx, item in enumerate(l):
            key, value = next(iter(item.items()))  # Extract key-value pair
            if beginning != 0 or idx > 0:
                f.write(",")  # Add a comma for valid JSON structure
            f.write(f'"{key}": {json.dumps(value)}')  # Write key-value directly

def merge(index1, index2, k,p):
    offset1 = 0
    offset2 = 0
    ans = []
    with open(f"index_table{p}.json", "a") as f:
            if index1 > 0 :
                f.write("{")
            else: 
                f.write("{")
    while True: 
        i, j = 0, 0
        firstEntry = offset1 + offset2
        buff1 = bringKObjects(k, index1, offset1,p)
        buff2 = bringKObjects(k, index2, offset2,p)
        if not buff1 and not buff2: 
            break
        if not buff1:
            writeToDisk(buff2, firstEntry,p )
            offset2 += k
            continue
        if not buff2: 
            writeToDisk(buff1, firstEntry, p)
            offset1 += k
            continue
        buff3 = []
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

        if i == len(buff1):
            offset1 += k
        elif j == len(buff2):
            offset2 += k
        ans.extend(buff3)
        writeToDisk(buff3, firstEntry,p)
    with open(f"index_table{p}.json", "a") as f:
        f.write("}")
    return ans


# # Example usage:
# result = merge(0, 1, 5)
# print(result)

def dumpObject(index,p):
    with open(f"index_table{p-1}.json", "r") as f:
        items = list(ijson.items(f, "item"))  # Convert to a list

    if index >= len(items):  #  invalid index
        return  

    with open(f"index_table{p}.json", "a") as f:
        json.dump([items[index]], f)  




def mergeAll(k, p):
    # Clear the file first
    with open(f"index_table{p}.json", "w") as op:
        op.write("[")
    
    with open(f"index_table{p-1}.json", "r") as f:
        items = list(ijson.items(f, "item"))
        
    lastIndex = len(items) - 1
    idx = 0
    
    # Process pairs
    while idx < lastIndex:
        merge(idx, idx + 1, k,p)
        idx += 2
        # Don't write comma after last pair
        if idx < lastIndex or (len(items) % 2 != 0):
            with open(f"index_table{p}.json", "a") as op:
                op.write(",")
    
    # Handle last odd item if exists
    if len(items) % 2 != 0:
        print("hello")
        with open(f"index_table{p}.json", "a") as f:
            json.dump(items[lastIndex], f)
    
    # Close the array
    with open(f"index_table{p}.json", "a") as op:
        op.write("]")

def final(number_of_chunks):
    result = math.ceil(math.log2(number_of_chunks))
    for i in range(result + 1):
        mergeAll(5, i + 1)
        

final(10)