from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import json
import re
from multiprocessing import Pool

# stopword = stopwords('english')

def due(filepath):
    middle = []
    result = []
    core = 64
    part = int(len(content) / core)
    p = Pool(core)
    try:
        for i in range(core):
            if i != core-1:
                start = i * part
                end = (i + 1) * part
            else:
                start = i * part
                end = len(content)
            result.append(p.apply_async(parallel, (start,end)))

        p.close()
        p.join()

        for item in result:
            res = item.get()
            middle += res

        # word_count = FreqDist(text)
        # word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))

        length = 0
        # 保存初始分词结果
        with open(filepath, 'w', encoding='utf-8') as fencifile:
            for i in tqdm(range(len(patent_number))):
                fencifile.write(json.dumps({'patent_number': patent_number[i], 'words': middle[i]}, ensure_ascii=False) + '\n')
                if len(middle[i]) > length:
                    length = len(middle[i])

        print(length)
    except Exception as e:
        print(e)


def parallel(start, end):
    # print('分词进度：')
    claim = []
    for num in tqdm(range(start, end)):
        # 分词
        words = word_tokenize(content[num])

        # lemmatizer = WordNetLemmatizer()
        # self.content[num] = [lemmatizer.lemmatize(word.lower()) for word in self.content[num]]
        # self.title[num] = [lemmatizer.lemmatize(word.lower()) for word in self.title[num]]

        # 提取词干
        stemmer = PorterStemmer()
        words = [stemmer.stem(word.lower()) for word in words]

        # 移除停用词
        claim.append([word for word in words if (word not in stopword and len(word) >= 2 and not re.match(r'[a-z]*[0-9]+[a-z]*', word))])

    return claim

# def get_conten(inputpath, outputpaht):
#     content = {}
#     with open(inputpath, 'r', encoding='utf-8'):
#         content.append(json.loads)

if __name__ == '__main__':
    content = []
    patent_number = []
    stopword = set(stopwords.words('english'))
    with open('../patent_inf/network/patent_content.txt', 'r', encoding='utf-8') as pc:
        for line in tqdm(pc):
            patent = json.loads(line)
            patent_number.append(patent['patent_number'])
            content.append(re.sub(r'[^a-zA-Z 0-9]+', ' ', patent['title'] + patent['abstract'] + patent['claim']))

    due('../patent_inf/network/train_fenci_all.txt')
