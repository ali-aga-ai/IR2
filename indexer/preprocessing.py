from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import json
from sortedcontainers import SortedSet

inverted_index = {}

nltk.download("wordnet")
nltk.download("stopwords")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

with open(r"C:\Documents\code\IR2\Assignment-data\bool_docs.json", "r") as f:
    data = json.load(f)

i = 1

for obj in data:
    words = obj["Abstract"].lower().split()  
    processed_words = [
        lemmatizer.lemmatize(stemmer.stem(word), pos="v")  
        for word in words if word not in stop_words  
    ]
    for word in processed_words:
        if word not in inverted_index:
            inverted_index[word] = SortedSet([i])
        else:
            inverted_index[word].add(i)
    obj["Abstract"] = " ".join(processed_words)  # reconstructing the abstract
    i+=1
    
for key,value in inverted_index.items(): # this is done because sorted set is a python object and cannot be convertedd to a "value" in json (aka non-serializable)
    inverted_index[key] = list(value)

with open("output.json","w") as f:
    f.write(json.dumps(inverted_index, indent =1))