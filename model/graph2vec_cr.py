import networkx as nx
import os
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from multiprocessing import Pool
from semantic_linking import semantic_linking
import logging
from semantic_linking import evaluate
from semantic_linking import cos_np
import numpy as np
from six.moves import range
from random import shuffle
from random import randrange
from random import choice
import matplotlib.pyplot as plt
import getinf
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEXT_MODEL_WORD2VEC = "word2vec"
TEXT_MODEL_DOC2VEC = "doc2vec"

class graph2vec():

    def __init__(self, text_model, dataset=None, walk_length=80, num_walks=10, size=200,
                 windows=10, negative=5, min_count=0, sg=1, hs=0, iter=5, workers=cpu_count()):
        global graph
        self.dataset = dataset
        self.text_model = text_model
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.size = size
        self.windows = windows
        self.negative = negative
        self.min_count = min_count
        self.sg = sg
        self.hs = hs
        self.iter = iter
        self.workers = workers
        self.e = evaluate()
        self.num2id = {}
        self.init_path()
        self.graph = nx.Graph()
        self.build_train_graph()
        self.graph = self.graph.to_undirected()
        print('length of nodes', len(self.graph.nodes()))
        print('length of edges', len(self.graph.edges))
        print(datetime.now())
        self.test_inf = getinf.GetInf('test')
        self.train_inf = getinf.GetInf('train')
        self.slm = semantic_linking(self.train_inf, self.test_inf, TEXT_MODEL_DOC2VEC, self.dataset)
        self.cn = cos_np(self.train_inf.train_number)

        try:
            self.model = Word2Vec.load(self.model_path)
        except:
            self.train_graph2vec()
            self.model = Word2Vec.load(self.model_path)

    def init_path(self):
        self.graph_path = '../patent_inf/PatentHIN.txt'
        self.model_path = '../patent_inf/model_save/deepwalk.model'
        self.semantic_path = '../patent_inf/network/train_semantic.txt'

    def get_semantic_links(self, train=True, k=30):
        if train:
            for candation_paper, semantic_links in self.slm.get_all_candidate_paper_semantic_linking(k=k):
                yield candation_paper, semantic_links
        else:
            for target_paper, semantic_links in self.slm.get_all_target_paper_semantic_linking(k=k):
                yield target_paper, semantic_links

    def build_train_graph(self):
        # G = nx.Graph()
        print(datetime.now())
        print('Build Train Graph')
        if os.path.exists(self.graph_path):
            i = 0
            with open(self.graph_path, encoding="utf8") as f:
                for l in f:
                    # if i < 200000:
                        adj = l.strip().split('\t')
                        self.graph.add_edge(*[adj[0][1:], adj[1][1:]])
                        self.graph.add_edge(*[adj[1][1:], adj[0][1:]])
                    #     i += 1
                    # else:
                    #     break

        print(len(self.graph.edges))

        with open(self.semantic_path, encoding='utf8') as f:
            i = 0
            for l in f:
                # if i < 200000:
                    adj = l.strip().split('\t')
                    self.graph.add_edge(*[adj[0][1:], adj[1][1:]])
                    self.graph.add_edge(*[adj[1][1:], adj[0][1:]])
                #     i += 1
                # else:
                #     break

        print(len(self.graph.edges))

        print('Graph Build Finish')
        print(datetime.now())

    def _save_train_graph(self, G):
        with open(self.graph_path + '_new', "w", encoding="utf8") as f:
            for node in tqdm(G.nodes()):
                for edge in G.neighbors(node):
                    f.write(node + "\t" + edge + "\n")

    def _get_a_a(self, authors):
        a_a = []
        while True:
            if len(authors) < 1:
                return a_a
            p = authors.pop()
            for q in authors:
                a_a.append([p, q])

    def _train(self, sentences):
        model = Word2Vec(sentences,
                              size=self.size,
                              window=self.windows,
                              negative=self.negative,
                              min_count=self.min_count,
                              sg=self.sg,
                              hs=self.hs,
                              iter=self.iter,
                              workers=self.workers,)
        model.save(self.model_path)

    def train_graph2vec(self):
        print('train_graph2vec')
        self.build_walks()

        class train_walks():

            def __init__(self, G):
                self.G = G
                self.nodes = G.nodes()

            def __iter__(self):
                with open('../patent_inf/model_save/walks.txt', 'r') as w:
                    for line in w:
                        yield eval(line.strip())
                    # for num_walk in range(num_walks):
                    #     print("random walk iteration: {} ".format(num_walk))
                    #     for node in nodes:
                    #         start = node
                    #         walk = []
                    #         while len(walk) != walk_length:
                    #             walk.append(start)
                    #             neighbors = self.G.neighbors(start)
                    #             neighbors = list(zip(range(neighbors.__length_hint__()), neighbors))
                    #             start = choice(neighbors)[1]
                    #         yield walk

        self._train(train_walks(self.graph))

    def update_graph(self, target_paper, semantic_link, old_a, new_a):
        authors = old_a + new_a
        p_a = [[target_paper, a] for a in authors]
        p_a_ = [[a, target_paper] for a in authors]
        a_a = [[au_au[0],au_au[1]] for au_au in self._get_a_a(authors)]
        a_a_ = [[au_au[1],au_au[0]] for au_au in self._get_a_a(authors)]
        p_sl = [[target_paper, sl] for sl in semantic_link]
        p_sl_ = [[sl, target_paper] for sl in semantic_link]
        self.graph.add_edges_from(p_a + a_a + p_sl + p_a_ + a_a_ + p_sl_)

    def _get_walk_from_one_nei(self, target_paper, semantic_linking, new_authors, old_authors, num_walks):
        walks = []
        walk = semantic_linking[:5] + [target_paper] + new_authors + old_authors + semantic_linking[5:]
        for _ in range(num_walks):
            shuffle(walk)
            walks.append(walk)
        return walks

    def _get_old_new_a(self, authors):
        old_a, new_a = [], []
        for a in authors:
            try:
                _ = self.model[a]
                old_a.append(a)
            except:
                new_a.append(a)
        return old_a, new_a

    def _get_random_walk_from_one_target_paper(self, walk_length, num_walks, target_paper, authors):

        nodes = [target_paper] + authors
        walks = []
        for _ in range(num_walks):
            shuffle(nodes)
            for node in nodes:
                start = node
                walk = []
                while True:
                    walk.append(start)
                    if len(walk) == walk_length:
                        walks.append(walk)
                        break
                    neighbors = self.graph.neighbors(start)
                    start = neighbors[randrange(len(neighbors))]
        return walks

    def graph2vec_cr(self, k=100, sl_num=10):
        i = 0
        candadite_paper_list = self.train_inf.train_number
        RECALL, MRR, NDCG, PRECSION = [0.] * len(self.e.k), \
                                      [0.] * len(self.e.k), \
                                      [0.] * len(self.e.k), \
                                      [0.] * len(self.e.k)

        t_syn0 = {}
        for node in self.model.wv.vocab.keys():
            t_syn0[node] = self.model[node]

        target_paper_list = self.test_inf.test_number
        target_paper_list_len = target_paper_list.__len__()

        for target_paper in tqdm(target_paper_list):
            semantic_links = self.slm._get_semantic_linking(target_paper, sl_num)
            authors = [a for a in self.test_inf.test_inventor[target_paper]]
            old_a, new_a = self._get_old_new_a(authors)
            self.update_graph(target_paper, semantic_links, old_a, new_a)
            walks = self._get_random_walk_from_one_target_paper(walk_length = 80,
                                                                num_walks = 10,
                                                                target_paper = target_paper,
                                                                authors = old_a + new_a)
            self.model.build_vocab(walks, update=True, min_count=0,)

            v = [0.] * self.model.vector_size
            for x in semantic_links + old_a:
                v += self.model[x]
            v = v / len(semantic_links + old_a)
            self.model.wv.syn0[self.model.wv.vocab[target_paper].index] = v
            for n_a in new_a:
                self.model.wv.syn0[self.model.wv.vocab[n_a].index] = v

            self.model.train(walks,
                             total_examples=self.model.corpus_count,
                             epochs=5,
                             start_alpha=0.001,
                             end_alpha=0.0001)

            matrix_B = []
            for candadite_paper in candadite_paper_list:
                matrix_B.append(self.model[candadite_paper])
            matrix_B = np.array(matrix_B)
            matrix_A = np.array([self.model[target_paper]])

            rec_result = self.cn.top_k(matrix_A, matrix_B, k=k)
            recall_all, mrr_all, precsion_all, ndcg_all = self.e.recall(rec_result, self.train_inf.train_citation[target_paper]), \
                                                          self.e.MRR(rec_result, self.train_inf.train_citation[target_paper]), \
                                                          self.e.precsion(rec_result, self.train_inf.train_citation[target_paper]), \
                                                          self.e.NDCG(rec_result, self.train_inf.train_citation[target_paper])

            print(target_paper, recall_all, len(self.test_inf.test_citation[target_paper]))
            RECALL, MRR, PRECSION, NDCG = np.sum([recall_all, RECALL], axis=0), \
                                          np.sum([mrr_all, MRR], axis=0), \
                                          np.sum([precsion_all, PRECSION], axis=0), \
                                          np.sum([ndcg_all, NDCG], axis=0)

            for node in self.model.wv.vocab.keys():
                try:
                    _ = t_syn0[node]
                    v = self.model[node] * 0.5 + t_syn0[node] * 0.5
                    self.model.wv.syn0[self.model.wv.vocab[node].index] = v
                except:
                    continue

            t_syn1 = {}
            for node in self.model.wv.vocab.keys():
                try:
                    _ = t_syn0[node]
                    v = t_syn0[node] * 0.3 + self.model[node] * 0.7
                    t_syn1[node] = v
                except:
                    t_syn1[node] = self.model[node]

            t_syn0 = t_syn1

            i += 1
            if i % 50 == 0:
                self.print_evaluate(self.e.k, RECALL / i,
                                    MRR / i,
                                    PRECSION / i,
                                    NDCG / i)

        self.print_evaluate(self.e.k, RECALL / target_paper_list_len,
                            MRR / target_paper_list_len,
                            PRECSION / target_paper_list_len,
                            NDCG / target_paper_list_len)

    def print_evaluate(self, k, RECALL, MRR, PRECSION, NDCG):

        k_recall,k_mrr,k_precsion,k_ndcg = dict(zip(k,RECALL)),dict(zip(k,MRR)),\
                                           dict(zip(k,PRECSION)),dict(zip(k,NDCG))
        for k_ in k:
            print("-------------------RECOMMENDATION NUMBER %d-----------------------------" %k_)
            print("RECALL: ", k_, "\t", k_recall[k_])
            print("PRECSION: ", k_, "\t", k_precsion[k_])
            print("MRR: ", k_, "\t", k_mrr[k_])
            print("NDCG: ", k_, "\t", k_ndcg[k_])

    # def process(self, min, max):
    #     walks = []
    #     for node in tqdm(self.nodes[min:max]):
    #         start = node
    #         walk = []
    #         while len(walk) != self.walk_length:
    #             walk.append(start)
    #             neighbors = self.graph.neighbors(start)
    #             neighbors = list(zip(range(neighbors.__length_hint__()), neighbors))
    #             start = choice(neighbors)[1]
    #         walks.append(walk)
    #     return walks

    def build_walks(self):
        nodes = self.graph.nodes()
        nodes = list(nodes)
        shuffle(nodes)
        graph = self.graph.neighbors
        part = int(len(nodes) / self.workers)
        walk_length = self.walk_length
        with open('../patent_inf/model_save/walks.txt', 'w') as w:
            for num_walk in range(self.num_walks):
                print("random walk iteration: {} ".format(num_walk))
                p = Pool(self.workers)
                walks = []
                result = []
                for i in range(self.workers):
                    if i != self.workers - 1:
                        result.append(p.apply_async(process, args=(nodes[i * part:(i + 1) * part], walk_length, graph)))
                    else:
                        result.append(p.apply_async(process, args=(nodes[i * part:len(nodes)], walk_length, graph)))

                p.close()
                p.join()
                for res in result:
                    walks += res.get()
                for item in tqdm(walks):
                    w.write(str(item) + '\n')


def process(nodes, walk_length, graph):
    walks = []
    for node in tqdm(nodes):
        start = node
        walk = []
        while len(walk) != walk_length:
            walk.append(start)
            neighbors = graph(start)  # neighbors(start)
            neighbors = list(zip(range(neighbors.__length_hint__()), neighbors))
            start = choice(neighbors)[1]
        walks.append(walk)
    return walks


if __name__ == '__main__':
    # words = getinf.GetInf().getWords()
    g2v = graph2vec(TEXT_MODEL_DOC2VEC)
    # g2v.train_graph2vec()
    g2v.graph2vec_cr(k=100, sl_num=10)

