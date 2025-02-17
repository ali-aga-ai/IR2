import ijson
import json 

def bringKObjects(k,index, offset): #index is basically konse waale se u wanna bring
    with open("index_table.json", "r") as f: 
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

def writeToDisk(l, beginning):
    with open("outputDisk.json", "a") as f:
        # Open the file and write the list as JSON, with commas separating each object
        for idx, item in enumerate(l):
                
            if beginning != 0 or idx > 0:
                f.write(",")  # Add a comma for subsequent items
            json.dump(item, f)

def merge(index1, index2, k):

    offset1 = 0
    offset2 = 0
    ans = []
    with open("outputDisk.json", "w") as f:
            f.write("[")
    while True: 
        i, j = 0, 0
        firstEntry = offset1 + offset2 # this indicates if first entry or not, if first entry dont add "," in the beginning of first term
        buff1 = bringKObjects(k, index1, offset1)  # List of dicts
        buff2 = bringKObjects(k, index2, offset2)
        if not buff1 and not buff2: 
            break
        if not buff1:
            writeToDisk(buff2,firstEntry)
            offset2+=k
            continue
        if not buff2: 
            writeToDisk(buff1,firstEntry)
            offset1+=k
            continue
        buff3 = []
        print(buff1)
        print(buff2)
        while i < len(buff1) and j < len(buff2):
            key1, val1 = next(iter(buff1[i].items()))  # Extract key-value pair from dict
            key2, val2 = next(iter(buff2[j].items()))

            if key1 < key2:
                buff3.append(buff1[i])
                i += 1
            elif key1 > key2:
                buff3.append(buff2[j])
                j += 1
            else:  # If keys are equal, merge their values
                merged_value = mergeLists(val1, val2)  # Assuming mergeLists() is defined
                buff3.append({key1: merged_value})
                i += 1
                j += 1

        if i == len(buff1):
            offset1+=k
        elif j == len(buff2):
            offset2+=k
        ans.extend(buff3)
        writeToDisk(buff3,firstEntry)
    with open("outputDisk.json", "a") as f:
        f.write("]")
    return ans

# Example usage:
result = merge(0, 1, 5)
print(result)
