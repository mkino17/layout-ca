import pickle
from pprint import pformat

with open("stack.pkl", "rb") as f:
    data = pickle.load(f)

print(data[0][0]['final_alignments'])
# aligned_pairs = data[0][0]['final_alignments']

# for pair in aligned_pairs:
    

with open("stack.txt", "w", encoding="utf-8") as f:
    f.write(pformat(data))

