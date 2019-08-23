import json
from tqdm import tqdm

print('start load....')
with open('../patent_inf/model_save/candidate.txt', 'r') as c:
    candidate = json.load(c)
print('end load...')

candidate1 = {}
for k, v in tqdm(candidate.items()):
    candidate1[k] = v[:30000]

with open('../patent_inf/model_save/candidate1.txt', 'w') as c1:
    json.dump(candidate1, c1)

