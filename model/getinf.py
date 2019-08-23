from collections import defaultdict
from tqdm import tqdm
import json


class GetInf:
    def __init__(self, flag=None):
        if flag == 'train':
            self.train_number = set()
            self.train_citation = defaultdict(list)
            self.train_inventor = defaultdict(list)
            self.train_assignee = defaultdict(list)

            self.getTrainAssignee()
            self.getTrainCitation()
            self.getTrainInventor()
            self.getTrainNumber()
        elif flag == 'test':
            self.test_number = set()
            self.test_citation = defaultdict(list)
            self.test_inventor = defaultdict(list)
            self.test_assignee = defaultdict(list)

            self.getTestAssignee()
            self.getTestCitation()
            self.getTestInventor()
            self.getTestNumber()
        else:
            pass

    def getTrainNumber(self):
        with open('../patent_inf/network/train_number2id.txt', 'r') as tn:
            for line in tn:
                self.train_number.add(line.strip().split('\t')[0])

    def getTestNumber(self):
        with open('../patent_inf/network/test_number.txt', 'r') as tn:
            for line in tn:
                self.test_number.add(line.strip())

    def getTrainCitation(self):
        with open('../patent_inf/network/train_citation.txt', 'r') as pc:
            for line in pc:
                p = line.strip().split('\t')
                self.train_citation[p[0][1:]].append(p[1][1:])

    def getTestCitation(self):
        with open('../patent_inf/network/test_citation.txt', 'r') as pc:
            for line in pc:
                p = line.strip().split('\t')
                self.test_citation[p[0][1:]].append(p[1][1:])

        # with open('../patent_inf/test_patent_inf_new.txt', 'r') as tpin:
        #     with open('../patent_inf/test_patent_inf_1.txt', 'w') as tpi:
        #         for line in tqdm(tpin):
        #             patent = json.loads(line)
        #             if patent['number'] in set(citation.keys()):
        #                 tpi.write(json.dumps(patent) + '\n')

    def getWords(self):
        pw = open('../patent_inf/network/train_fenci.txt', 'r')
        return pw

    def getPatents(self):
        p = open('../patent_inf/network/patent_content.txt', 'r')
        return p

    def getTrainInventor(self):
        with open('../patent_inf/network/train_inventor.txt', 'r') as pc:
            for line in pc:
                p = line.strip().split('\t')
                self.train_inventor[p[0][1:]].append(p[1][1:])

    def getTestInventor(self):
        with open('../patent_inf/network/test_inventor.txt', 'r') as pc:
            for line in pc:
                p = line.strip().split('\t')
                self.test_inventor[p[0][1:]].append(p[1][1:])

    def getTrainAssignee(self):
        with open('../patent_inf/network/train_assignee.txt', 'r') as pc:
            for line in pc:
                p = line.strip().split('\t')
                self.train_assignee[p[0][1:]].append(p[1][1:])

    def getTestAssignee(self):
        with open('../patent_inf/network/test_assignee.txt', 'r') as pc:
            for line in pc:
                p = line.strip().split('\t')
                self.test_assignee[p[0][1:]].append(p[1][1:])


if __name__ == '__main__':
    a = GetInf()
