import logging
import getinf
from tqdm import tqdm
import json
import numpy as np
from gensim.models import doc2vec, word2vec
from BM25 import BM25
import random
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from bert import modeling
import tensorflow as tf
from multiprocessing import cpu_count
from multiprocessing import Pool
import math
from bert_serving.client import BertClient
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class doc2vec_cr():

    def __init__(self, train_inf, test_inf, dataset=None, vector_size=300, window=5, min_count=5, sample=1e-3,
               negative=5, epochs=5, dm=0, hs=1, dm_concat=1, dbow_words=1, workers=cpu_count()):
        self.dataset = dataset
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.negative = negative
        self.epochs = epochs
        self.dm = dm
        self.hs = hs
        self.dm_concat = dm_concat
        self.dbow_words = dbow_words
        self.workers = workers
        self.e = evaluate()
        self.train_inf = train_inf
        self.test_inf = test_inf
        self.number = self.train_inf.train_number
        self.Citation = self.test_inf.test_citation
        self.cn = cos_np(self.number)
        
        try:
            self.model = doc2vec.Doc2Vec.load('../patent_inf/model_save/doc2vec.model')
        except:
            self.train_doc2vec()
            self.model = doc2vec.Doc2Vec.load('../patent_inf/model_save/doc2vec.model')

    def _train(self, corpus):
        model = doc2vec.Doc2Vec(vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  sample=self.sample,
                                  negative=self.negative,
                                  epochs=self.epochs,
                                  dm=self.dm,
                                  hs=self.hs,
                                  dm_concat=self.dm_concat,
                                  dbow_words=self.dbow_words,
                                  workers=self.workers)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save('../patent_inf/model_save/doc2vec.model')
        model.save_word2vec_format('../patent_inf/model_save/word2vec.model', binary=False)

    def train_doc2vec(self):
        x_train = []
        for line in self.dataset:
            patent = json.loads(line)
            x_train.append(doc2vec.TaggedDocument(patent['words'], [patent['patent_number']]))

        self._train(x_train)

    def get_vector(self, id=None, text=None):
        if id == None:
            raise ValueError("id is none!")
        if id in self.number:
            return self.model.docvecs[id]
        return self.model.infer_vector(text)

    def init_matrix_B(self):
        matrix_B = []
        for candadite_paper in self.number:
            matrix_B.append(self.get_vector(candadite_paper))
        matrix_B = np.array(matrix_B)
        return matrix_B

    def get_sim_top_k(self, k, id=None, text=None,):
        matrix_A = self.get_vector(id=id, text=text)
        matrix_A = [matrix_A]
        patent = []
        # return self.cn.top_k(matrix_A, self.matrix_B, k=k)
        for sim in self.model.docvecs.most_similar(matrix_A, topn=k):
            patent.append(sim[0])

        return patent

    def citation_rec(self, k=30000):
        self.matrix_B = self.init_matrix_B()
        RECALL, MRR, NDCG, PRECSION = [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(self.e.k)

        target_paper_list = list(self.target_paper.keys())
        target_paper_list_len = target_paper_list.__len__()
        i = 0
        result = []
        for target_paper in tqdm(target_paper_list[:1000]):
            rec_result = self.get_sim_top_k(k, id=target_paper, text=self.target_paper[target_paper])
            citation = self.Citation[target_paper]
            # print(rec_result)
            # if len(citation) >= 10:
            #     i += 1
            recall_all, mrr_all, precsion_all, ndcg_all = self.e.recall(rec_result, citation), \
                                                          self.e.MRR(rec_result, citation), \
                                                          self.e.precsion(rec_result, citation), \
                                                          self.e.NDCG(rec_result, citation)
            result.append(rec_result)
            # recall_all = metrics.recall_score(rec_result, citation)
            # precsion_all = metrics.precision_score(rec_result, citation)
            # ndcg_all = self.e.NDCG(rec_result, self.Citation[target_paper])
            # mrr_all = self.e.MRR(rec_result, self.Citation[target_paper])
            # print(target_paper, recall_all)
            RECALL, MRR, PRECSION, NDCG = np.sum([recall_all, RECALL], axis=0), \
                                          np.sum([mrr_all, MRR], axis=0), \
                                          np.sum([precsion_all, PRECSION], axis=0), \
                                          np.sum([ndcg_all, NDCG], axis=0)
        # with open('../patent_inf/model_save/candidate.txt', 'w') as c:
        #     json.dump(dict(zip(target_paper_list, result)), c, indent=4)
        target_paper_list_len = 1000
        self.print_evaluate(self.e.k, RECALL / target_paper_list_len, MRR / target_paper_list_len,
                            PRECSION / target_paper_list_len, NDCG / target_paper_list_len)
        # print('number of citations lager than 10:', i)

    def print_evaluate(self,k,RECALL, MRR, PRECSION, NDCG):

        k_recall,k_mrr,k_precsion,k_ndcg = dict(zip(k,RECALL)),dict(zip(k,MRR)), dict(zip(k,PRECSION)),dict(zip(k,NDCG))

        for k_ in k:
            print("-------------------RECOMMENDATION NUMBER %d-----------------------------" %k_)
            print("RECALL: ", k_, "\t", k_recall[k_])
            print("PRECSION: ", k_, "\t", k_precsion[k_])
            print("MRR: ", k_, "\t", k_mrr[k_])
            print("NDCG: ", k_, "\t", k_ndcg[k_])

    def get_target(self):
        self.target_paper = {}
        with open('../patent_inf/network/test_fenci.txt', 'r') as tf:
            for line in tf:
                patent = json.loads(line)
                self.target_paper[patent['patent_number']] = patent['words']


class word2vec_cr():
    def __init__(self, train_inf, test_inf, dataset=None, vector_size=300, window=5, min_count=5, sample=1e-3,
                 negative=5, epochs=5, dm=0, hs=1, dm_concat=1, dbow_words=1, workers=cpu_count()):
        self.dataset = dataset
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sample = sample
        self.negative = negative
        self.epochs = epochs
        self.dm = dm
        self.hs = hs
        self.dm_concat = dm_concat
        self.dbow_words = dbow_words
        self.workers = workers
        self.e = evaluate()
        self.train_inf = train_inf
        self.test_inf = test_inf
        self.number = self.train_inf.train_number
        self.Citation = self.test_inf.test_citation
        self.cn = cos_np(self.number)
        self.w_embedding = []

        try:
            self.p_embedding = np.load('../patent_inf/model_save/p_embedding.npy')
        except:
            try:
                self.w2v = doc2vec.Doc2Vec.load('../patent_inf/model_save/word2vec.model')
            except:
                print('No word2vec model!!!!')
                exit(1)

            try:
                self.tfidf = np.load('../patent_inf/model_save/tfidf.npy')
            except:
                self._tfidf()
                self.tfidf = np.load('../patent_inf/model_save/tfidf.npy')

                with open('../patent_inf/model_save/wordlist.txt', 'r') as w:
                    for line in w:
                        self.w_embedding.append(self.w2v.wv[line.strip()])
                self.p_embedding = np.dot(self.tfidf, self.w_embedding)
                np.save('../patent_inf/model_save/p_embedding', self.p_embedding)

    def _train(self, corpus):
        model = doc2vec.Doc2Vec(vector_size=self.vector_size,
                                window=self.window,
                                min_count=self.min_count,
                                sample=self.sample,
                                negative=self.negative,
                                epochs=self.epochs,
                                dm=self.dm,
                                hs=self.hs,
                                dm_concat=self.dm_concat,
                                dbow_words=self.dbow_words,
                                workers=self.workers)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save('../patent_inf/model_save/doc2vec.model')
        model.save_word2vec_format('../patent_inf/model_save/word2vec.model', binary=False)

    def train_doc2vec(self):
        x_train = []
        for line in self.dataset:
            patent = json.loads(line)
            x_train.append(doc2vec.TaggedDocument(patent['words'], [patent['patent_number']]))

        self._train(x_train)

    def _tfidf(self):
        x_train = []
        # 将文本中的词语转换为词频矩阵
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        for line in self.dataset:
            patent = json.loads(line)
            x_train.append([patent['words']])
            # self.number.append(patent['patent_number'])
        # 计算个词语出现的次数
        X = vectorizer.fit_transform(x_train)
        words = vectorizer.get_feature_names()
        # 将词频矩阵X统计成TF-IDF值
        tfidf = transformer.fit_transform(X)
        weight = tfidf.toarray()
        np.save('../patent_inf/model_save/tfidf', weight)
        with open('../patent_inf/model_save/wordlist.txt', 'w') as w:
            for word in words:
                w.write(word + '\n')

    def get_vector(self, id=None, text=None):
        if id == None:
            raise ValueError("id is none!")
        if id in self.number:
            return self.p_embedding[id]
        return self.model.infer_vector(text)

    def init_matrix_B(self):
        matrix_B = []
        for candadite_paper in self.number:
            matrix_B.append(self.get_vector(candadite_paper))
        matrix_B = np.array(matrix_B)
        return matrix_B

    def get_sim_top_k(self, k, id=None, text=None, ):
        matrix_A = self.get_vector(id=id, text=text)
        matrix_A = [matrix_A]
        patent = []
        # return self.cn.top_k(matrix_A, self.matrix_B, k=k)
        for sim in self.model.docvecs.most_similar(matrix_A, topn=k):
            patent.append(sim[0])

        return patent

    def citation_rec(self, k=30000):
        self.matrix_B = self.init_matrix_B()
        RECALL, MRR, NDCG, PRECSION = [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(
            self.e.k)

        target_paper_list = list(self.target_paper.keys())
        target_paper_list_len = target_paper_list.__len__()
        i = 0
        result = []
        for target_paper in tqdm(target_paper_list[:1000]):
            rec_result = self.get_sim_top_k(k, id=target_paper, text=self.target_paper[target_paper])
            citation = self.Citation[target_paper]
            # print(rec_result)
            # if len(citation) >= 10:
            #     i += 1
            recall_all, mrr_all, precsion_all, ndcg_all = self.e.recall(rec_result, citation), \
                                                          self.e.MRR(rec_result, citation), \
                                                          self.e.precsion(rec_result, citation), \
                                                          self.e.NDCG(rec_result, citation)
            result.append(rec_result)
            # recall_all = metrics.recall_score(rec_result, citation)
            # precsion_all = metrics.precision_score(rec_result, citation)
            # ndcg_all = self.e.NDCG(rec_result, self.Citation[target_paper])
            # mrr_all = self.e.MRR(rec_result, self.Citation[target_paper])
            # print(target_paper, recall_all)
            RECALL, MRR, PRECSION, NDCG = np.sum([recall_all, RECALL], axis=0), \
                                          np.sum([mrr_all, MRR], axis=0), \
                                          np.sum([precsion_all, PRECSION], axis=0), \
                                          np.sum([ndcg_all, NDCG], axis=0)
        # with open('../patent_inf/model_save/candidate.txt', 'w') as c:
        #     json.dump(dict(zip(target_paper_list, result)), c, indent=4)
        target_paper_list_len = 1000
        self.print_evaluate(self.e.k, RECALL / target_paper_list_len, MRR / target_paper_list_len,
                            PRECSION / target_paper_list_len, NDCG / target_paper_list_len)
        # print('number of citations lager than 10:', i)

    def print_evaluate(self, k, RECALL, MRR, PRECSION, NDCG):

        k_recall, k_mrr, k_precsion, k_ndcg = dict(zip(k, RECALL)), dict(zip(k, MRR)), dict(zip(k, PRECSION)), dict(
            zip(k, NDCG))

        for k_ in k:
            print("-------------------RECOMMENDATION NUMBER %d-----------------------------" % k_)
            print("RECALL: ", k_, "\t", k_recall[k_])
            print("PRECSION: ", k_, "\t", k_precsion[k_])
            print("MRR: ", k_, "\t", k_mrr[k_])
            print("NDCG: ", k_, "\t", k_ndcg[k_])

    def get_target(self):
        self.target_paper = {}
        with open('../patent_inf/network/test_fenci.txt', 'r') as tf:
            for line in tf:
                patent = json.loads(line)
                self.target_paper[patent['patent_number']] = patent['words']


class bm25_cr():
    def __init__(self, train_inf, test_inf, dataset=None, workers=cpu_count()):
        self.dataset = dataset
        self.workers = workers
        self.e = evaluate()
        self.train_inf = train_inf
        self.test_inf = test_inf
        self.number = []
        self.Citation = self.test_inf.test_citation

        try:
            bm = open('../patent_inf/model_save/bm25.model', 'r')
            parm = json.load(bm)
        except:
            self.train_bm25()
            bm = open('../patent_inf/model_save/bm25.model', 'r')
            parm = json.load(bm)

        bm.close()
        self.model = BM25()
        self.model.f = parm['f']
        self.model.idf = parm['idf']
        self.model.avgdl = parm['avgdl']
        self.average_idf = parm['average_idf']
        self.model.corpus_size = parm['corpus_size']
        self.model.number = parm['number']
        self.model.length = parm['length']
        del parm

        self.cn = cos_np(self.number)

    def _train(self, corpus):
        model = BM25(corpus, self.workers)
        model.initialize()
        average_idf = sum(map(lambda k: float(model.idf[k]), model.idf.keys())) / len(model.idf.keys())
        # print(model.number)
        with open('../patent_inf/model_save/bm25.model', 'w') as bm:
            json.dump({'idf': model.idf, 'f': model.f, 'corpus_size': model.corpus_size, 'average_idf': average_idf,
                       'avgdl': model.avgdl, 'number': model.number, 'length': model.length}, bm)

    def train_bm25(self):
        self._train(self.dataset)

    def citation_rec(self, k=50000):
        RECALL, MRR, NDCG, PRECSION = [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(
            self.e.k)
        try:
            target_paper_list = []
            rec_result = []
            print('start load...')
            with open('../patent_inf/model_save/candidate.txt', 'r') as c:
                candidate = json.load(c)
            print('end load...')

            for k, v in candidate.items():
                target_paper_list.append(k)
                rec_result.append(v)

            target_paper_list_len = target_paper_list.__len__()
        except:
            print('start citation_rec')

            target_paper_list = list(self.target_paper.keys())
            target_paper_list = random.sample(target_paper_list, 1000)
            target_paper_list_len = target_paper_list.__len__()

            part = int(target_paper_list_len / self.workers)
            p = Pool(self.workers)
            result = []
            ave = self.average_idf
            m = self.model
            for i in range(self.workers):
                content = []
                if i != self.workers-1:
                    for target in target_paper_list[i * part:(i + 1) * part]:
                        content.append(self.target_paper[target])
                else:
                    for target in target_paper_list[i * part:]:
                        content.append(self.target_paper[target])
                result.append(p.apply_async(process, args=(m, content, ave, k,)))

            p.close()
            p.join()

            rec_result = []

            for res in result:
                rec_result += res.get()

            with open('../patent_inf/model_save/candidate.txt', 'w') as c:
                json.dump(dict(zip(target_paper_list, rec_result)), c)

        # index = random.sample(list(range(20971)), 1000)
        index = range(1000)
        for i in tqdm(index):
            citation = self.Citation[target_paper_list[i]]
            # print(len(paper))
            recall_all, mrr_all, precsion_all, ndcg_all = self.e.recall(rec_result[i], citation), \
                                                          self.e.MRR(rec_result[i], citation), \
                                                          self.e.precsion(rec_result[i], citation), \
                                                          self.e.NDCG(rec_result[i], citation)

            RECALL, MRR, PRECSION, NDCG = np.sum([recall_all, RECALL], axis=0), \
                                          np.sum([mrr_all, MRR], axis=0), \
                                          np.sum([precsion_all, PRECSION], axis=0), \
                                          np.sum([ndcg_all, NDCG], axis=0)

        target_paper_list_len = 1000
        self.print_evaluate(self.e.k, RECALL / target_paper_list_len, MRR / target_paper_list_len,
                            PRECSION / target_paper_list_len, NDCG / target_paper_list_len)
        '''for paper, target in tqdm(zip(rec_result_1, target_paper_list_1)):
            citation = self.Citation[target]
            # print(len(paper))
            recall_all, mrr_all, precsion_all, ndcg_all = self.e.recall(paper, citation), \
                                                          self.e.MRR(paper, citation), \
                                                          self.e.precsion(paper, citation), \
                                                          self.e.NDCG(paper, citation)

            # if len(citation) >= 10:
            #     i += 1
            # recall_all = metrics.recall_score(paper, citation)
            # precsion_all = metrics.precision_score(paper, citation)
            # ndcg_all = self.e.NDCG(paper, self.Citation[target])
            # mrr_all = self.e.MRR(paper, self.Citation[target])
            # print(target, recall_all)
            RECALL, MRR, PRECSION, NDCG = np.sum([recall_all, RECALL], axis=0), \
                                                np.sum([mrr_all, MRR], axis=0), \
                                                np.sum([precsion_all, PRECSION], axis=0), \
                                                np.sum([ndcg_all, NDCG], axis=0)

        self.print_evaluate(self.e.k, RECALL / target_paper_list_len, MRR / target_paper_list_len,
                            PRECSION / target_paper_list_len, NDCG / target_paper_list_len)
        # self.print_evaluate(self.e.k, RECALL / i, MRR / i, PRECSION / i, NDCG / i)'''


    def print_evaluate(self, k, RECALL, MRR, PRECSION, NDCG):

        k_recall, k_mrr, k_precsion, k_ndcg = dict(zip(k, RECALL)), dict(zip(k, MRR)), dict(zip(k, PRECSION)), dict(
            zip(k, NDCG))

        for k_ in k:
            print("-------------------RECOMMENDATION NUMBER %d-----------------------------" % k_)
            print("RECALL: ", k_, "\t", k_recall[k_])
            print("PRECSION: ", k_, "\t", k_precsion[k_])
            print("MRR: ", k_, "\t", k_mrr[k_])
            print("NDCG: ", k_, "\t", k_ndcg[k_])

    def get_target(self):
        self.target_paper = {}
        with open('../patent_inf/network/test_fenci.txt', 'r') as tf:
            for line in tf:
                patent = json.loads(line)
                self.target_paper[patent['patent_number']] = patent['words']


class BERT():
    def __init__(self, train_inf, test_inf, dataset=None, workers=cpu_count()):
        self.dataset = dataset
        self.workers = workers
        self.e = evaluate()
        self.train_inf = train_inf
        self.test_inf = test_inf
        self.number = []
        self.Citation = self.test_inf.test_citation
        # self.model = BertClient()
        self.vec = {}

        try:
            with open('../patent_inf/model_save/bert.model', 'r') as b:
                for line in b:
                    p = line.strip().split('\t')
                    self.vec[p[0]] = eval(p[1])
        except:
            self.train_bert()
            with open('../patent_inf/model_save/bert.model', 'r') as b:
                for line in b:
                    p = line.strip().split('\t')
                    self.vec[p[0]] = eval(p[1])

        self.number = set(self.vec.keys())
        self.cn = cos_np(self.number)

    # def _train(self, corpus):
    #     bc = BertClient()
    #     sentence = []
    #     number = []
    #     i = 0
    #     for line in tqdm(corpus):
    #         if i < 10:
    #             p = json.loads(line.strip())
    #             sentence.append(p['title'] + p['abstract'])
    #             number.append(p['patent_number'])
    #             i += 1
    #         else:
    #             break
    #     vec = bc.encode(sentence)
    #     rows, cols = vec.shape
    #     with open('../patent_inf/model_save/bert.model', 'w') as b:
    #         for i in range(rows):
    #             b.write(number[i] + '\t' + str(vec[i].tolist()) + '\n')
    #     # np.save('../patent_inf/model_save/bert', vec)

    def _train(self, corpus):
        # 这里是下载下来的bert配置文件
        pathname = "multi_cased_L-12_H-768_A-12/bert_model.ckpt"  # 模型地址
        #  创建bert的输入
        bert_config = modeling.BertConfig.from_json_file("multi_cased_L-12_H-768_A-12/bert_config.json")  # 配置文件地址。
        configsession = tf.ConfigProto()
        configsession.gpu_options.allow_growth = True
        sess = tf.Session(config=configsession)
        input_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_ids")
        input_mask = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="input_mask")
        segment_ids = tf.placeholder(shape=[64, 128], dtype=tf.int32, name="segment_ids")

        with sess.as_default():
            self.model = modeling.BertModel(
                config=bert_config,
                is_training=True,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=False)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())  # 这里尤其注意，先初始化，在加载参数，否者会把bert的参数重新初始化。这里和demo1是有区别的
            saver.restore(sess, pathname)
            print('load model sucessful')

    def train_bert(self):
        self._train(self.dataset)

    def get_vector(self, id=None):
        if id == None:
            raise ValueError("id is none!")
        if id in self.number:
            return self.vec[id]
        return self.target_vec[id]

    def init_matrix_B(self):
        matrix_B = []
        for candadite_paper in self.number:
            matrix_B.append(self.get_vector(candadite_paper))
        matrix_B = np.array(matrix_B)
        return matrix_B

    def get_sim_top_k(self, k, id=None):
        matrix_A = self.get_vector(id=id)
        matrix_A = [matrix_A]
        patent = []
        return self.cn.top_k(matrix_A, self.matrix_B, k=k)
        # for sim in self.cn.top_k(matrix_A, topn=k):
        # patent.append(sim[0])
        # return patent

    def citation_rec(self, k=30000):
        self.matrix_B = self.init_matrix_B()
        RECALL, MRR, NDCG, PRECSION = [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(self.e.k), [0.] * len(
            self.e.k)

        target_paper_list = list(self.target_paper.keys())
        target_paper_list_len = target_paper_list.__len__()

        for target_paper in tqdm(target_paper_list):
            rec_result = self.get_sim_top_k(k, id=target_paper)
            citation = self.Citation[target_paper]
            # print(rec_result)
            recall_all, mrr_all, precsion_all, ndcg_all = self.e.recall(rec_result, self.Citation[target_paper]), \
                                                          self.e.MRR(rec_result, self.Citation[target_paper]), \
                                                          self.e.precsion(rec_result, self.Citation[target_paper]), \
                                                          self.e.NDCG(rec_result, self.Citation[target_paper])

            # recall_all = metrics.recall_score(rec_result, citation)
            # precsion_all = metrics.precision_score(rec_result, citation)
            # ndcg_all = self.e.NDCG(rec_result, self.Citation[target_paper])
            # mrr_all = self.e.MRR(rec_result, self.Citation[target_paper])
            print(target_paper, recall_all)
            RECALL, MRR, PRECSION, NDCG = np.sum([recall_all, RECALL], axis=0), \
                                          np.sum([mrr_all, MRR], axis=0), \
                                          np.sum([precsion_all, PRECSION], axis=0), \
                                          np.sum([ndcg_all, NDCG], axis=0)

        self.print_evaluate(self.e.k, RECALL / target_paper_list_len, MRR / target_paper_list_len,
                            PRECSION / target_paper_list_len, NDCG / target_paper_list_len)

    def print_evaluate(self, k, RECALL, MRR, PRECSION, NDCG):

        k_recall, k_mrr, k_precsion, k_ndcg = dict(zip(k, RECALL)), dict(zip(k, MRR)), dict(zip(k, PRECSION)), dict(
            zip(k, NDCG))

        for k_ in k:
            print("-------------------RECOMMENDATION NUMBER %d-----------------------------" % k_)
            print("RECALL: ", k_, "\t", k_recall[k_])
            print("PRECSION: ", k_, "\t", k_precsion[k_])
            print("MRR: ", k_, "\t", k_mrr[k_])
            print("NDCG: ", k_, "\t", k_ndcg[k_])

    def get_target(self):
        bc = BertClient()
        self.target_paper = {}
        number = set()
        content = []
        with open('../patent_inf/network/test_content.txt', 'r') as tf:
            for line in tf:
                patent = json.loads(line)
                self.target_paper[patent['patent_number']] = patent['title'] + patent['abstract']
                number.add(patent['patent_number'])
                content.append(patent['title'] + patent['abstract'])

        self.target_vec = bc.encode(content)
        # rows, cols = vec.shape


class cos_np:

    def __init__(self, candidate_number):
        self.cp_list = candidate_number

    def compute_cos(self, matrix_A, matrix_B):
        matrix_A_norm = np.sqrt(np.multiply(matrix_A, matrix_A).sum(axis=1)).reshape(len(matrix_A),1)
        matrix_B_norm = np.sqrt(np.multiply(matrix_B, matrix_B).sum(axis=1)).reshape(len(matrix_B),1)
        matrix_A_matrix_B = np.matmul(matrix_A, matrix_B.transpose())
        matrix_A_norm_matrix_B_norm = np.matmul(matrix_A_norm, matrix_B_norm.transpose())
        return np.divide(matrix_A_matrix_B, matrix_A_norm_matrix_B_norm)[0]

    def top_k(self, matrix_A, matrix_B, k=10):
        return [x[0] for x in sorted(list(zip(self.cp_list, self.compute_cos(matrix_A, matrix_B))),key=lambda x:x[1],reverse=True)][:k]


class evaluate():

    def __init__(self):
        self.k = [300000]    # [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    def recall(self, rec_l, citation_l):
        recall_all = []
        for k in self.k:
            new_rec_l = rec_l[:k]
            recall_all.append(len(set(new_rec_l) & set(citation_l)) / len(citation_l))
        return recall_all

    def NDCG(self, rec_l, citation_l):
        NDCG_all = []
        for k in self.k:
            new_rec_l = rec_l[:k]
            rr_l = list(set(new_rec_l) & set(citation_l))
            if len(rr_l) == 0:
                NDCG_all.append(0.)
                continue
            dcg = 0.
            idcg = 1.
            for i in range(1, len(rr_l)):
                idcg += 1 / math.log(i + 1, 2)
            if rec_l[0] in citation_l:
                dcg = 1.
            for i in range(1, len(rec_l)):
                if rec_l[i] in citation_l:
                    dcg += 1 / math.log(i + 1, 2)
            NDCG_all.append(dcg / idcg)
        return NDCG_all

    def MRR(self, rec_l, citation_l):
        mrr_all = []
        for k in self.k:
            new_rec_l = rec_l[:k]
            H = False
            for each_rec_l in new_rec_l:
                if each_rec_l in citation_l:
                    H = True
                    mrr_all.append(1 / (new_rec_l.index(each_rec_l) + 1))
                    break
            if not H:
                mrr_all.append(0.)
        return mrr_all

    def precsion(self, rec_l, citation_l):
        precsion_all = []
        for k in self.k:
            new_rec_l = rec_l[:k]
            precsion_all.append(len(set(new_rec_l) & set(citation_l)) / len(new_rec_l))
        return precsion_all


class semantic_linking():
    def __init__(self, train_inf, test_inf, text_model, dataset=None):
        self.dataset = dataset
        self.text_model = text_model
        self.train_inf = train_inf
        self.test_inf = test_inf
        if self.text_model == "word2vec":
            self.model = word2vec_cr(train_inf,test_inf,self.dataset)
        elif self.text_model == "doc2vec":
            self.model = doc2vec_cr(train_inf,test_inf,self.dataset)
        self.model.matrix_B = self.model.init_matrix_B()

    def _get_semantic_linking(self, paper, k, text=None):
        if paper in self.test_inf.test_number:
            return self.model.get_sim_top_k(k=k, id=paper, text=text)
        return self.model.get_sim_top_k(k=k, id=paper)

    def get_all_candidate_paper_semantic_linking(self, k):
        print('get_all_candidate_paper_semantic_linking')
        for candidate_paper in tqdm(self.train_inf.train_number):
            yield candidate_paper, self._get_semantic_linking(candidate_paper, k)

    def get_all_target_paper_semantic_linking(self, k):
        print('get_all_target_paper_semantic_linking')
        with open('../patent_inf/network/test_fenci.txt', 'r') as tf:
            for line in tf:
                patent = json.loads(line)
                yield patent['patent_number'], self._get_semantic_linking(paper=patent['patent_number'], k=k, text=patent['words'])


def process(client, content, ave, k):
    return client.get_scores(content, ave, k)

if __name__ == '__main__':
    words = getinf.GetInf().getWords()
    content = getinf.GetInf().getPatents()
    train_inf = getinf.GetInf('train')
    test_inf = getinf.GetInf('test')
    # model = doc2vec_cr(train_inf, test_inf, words)
    # model.get_target()
    # model.citation_rec()
    model = bm25_cr(train_inf, test_inf, words)
    model.get_target()
    model.citation_rec()
    # model = BERT(train_inf, test_inf, content)
    # model.get_target()
    # model.citation_rec()

