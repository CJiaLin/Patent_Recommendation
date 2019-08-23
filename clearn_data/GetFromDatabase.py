import pandas as pd
from tqdm import tqdm
from random import sample
from multiprocessing import Pool, cpu_count
import json
from h5py_data_cr import citation_h5py_dataset
from pymongo import MongoClient
from bson import json_util


class get_data():
    def __init__(self):
        self.target_lines = target_lines

    def GetInf(self, inputfile, output, test=True):
        number = set()
        citation = set()
        if test:
            name_content = 'test_patent_content.txt'
            name_other = 'test_patent_other.txt'
            name_num = 'test_number.txt'
        else:
            name_content = 'train_patent_content.txt'
            name_other = 'train_patent_other.txt'
            name_num = 'train_number.txt'

        with open(inputfile, 'r', encoding='utf-8') as inputf:
            for line in tqdm(inputf):
                patent = json.loads(line.strip('\n').strip(','))
                if (len(patent['citation']) > 0) and (len(patent['inventor']) > 0) and (len(patent['assignee']) > 0) \
                        and 'title' in set(patent.keys()):
                    patent['inventor'] = self._deef_inf(patent['inventor'], 'id')
                    patent['assignee'] = self._deef_inf(patent['assignee'], 'id')
                    cite = self._deef_inf(patent['citation'], 'citation_id')
                    patent['citation'] = cite
                    if test:
                        citation.update(set(cite))
                    number.add(patent['number'])
                    self.target_lines.append(patent)
                else:
                    continue
        print("Saving content......")
        with open(output + name_content, 'w', encoding='utf-8') as outinf:
            for patent in tqdm(self.target_lines):
                outinf.write(json.dumps({'number': patent['number'], 'title': patent['title'],
                                         'abstract': patent['abstract'], 'claim': patent['claim']}) + '\n')
        print("Saving other......")
        with open(output + name_other, 'w', encoding='utf-8') as outinf:
            for patent in tqdm(self.target_lines):
                outinf.write(json.dumps({'number': patent['number'], 'ipcr': patent['ipcr'],
                                         'inventor': patent['inventor'], 'assignee': patent['assignee'],
                                         'citation': patent['citation']}) + '\n')
        print("Saving num......")
        with open(output + name_num, 'w', encoding='utf-8') as outnum:
            for num in number:
                outnum.write(num + '\n')

        # if test:
        #     with open(output + 'test_citation_number.txt', 'w', encoding='utf-8') as tcn:
        #         for num in citation:
        #             tcn.write(num + '\n')

    def _deef_inf(self, items, name):
        ret = []
        for item in items:
            ret.append(item[name])

        return ret

    def _get_test_citation(self, inputfile):
        self.test_citation = []
        with open(inputfile, 'r', encoding='utf-8') as inputf:
            for line in tqdm(inputf):
                self.test_citation.append(line.strip())

    def GetTrain(self, inputfile, outputfile):
        self.core = cpu_count()
        self._get_test_citation(inputfile)

        part = int(len(self.test_citation) / self.core)
        for i in range(self.core):
            if i != self.core-1:
                start = part * i
                end = part * (i + 1)
            else:
                start = part * i
                end = len(self.test_citation)
            self._get_from_database(start, end, outputfile)

    def _get_from_database(self, start, end, outputfile):
        client = MongoClient('210.45.215.233', 9001)
        db = client['uspto_patent']
        collection = db['patent_all']
        # patent_inf = []
        search_result = collection.find({'number': {"$in": self.test_citation[start:end]}, 'abstract': {'$exists': 'true'}})
        with open(outputfile + 'uspto_train_patent.json', 'a', encoding='utf-8') as outfile:
            for patent in tqdm(search_result):
                outfile.write(json_util.dumps(patent) + '\n')
        # return patent_inf

    def Select(self, inputfile, chd):
        self.chd = chd
        train_number = set()
        with open(inputfile + 'train_num.txt', 'r', encoding='utf-8') as tn:
            for line in tn:
                train_number.add(line.strip())
        test_number = set()
        with open(inputfile + 'test_number.txt', 'r', encoding='utf-8') as ten:
            for line in ten:
                test_number.add(line.strip())

        test_subset = set()
        train_subset = set()
        inventors = set()
        assignees = set()
        ipcrs = set()
        test_other = []
        with open(inputfile + 'test_patent_other.txt', 'r', encoding='utf-8') as tpo:
            for line in tqdm(tpo):
                patent = json.loads(line.strip())
                delete = set()
                for cite in patent['citation']:
                    if cite not in train_number or cite in test_number:
                        delete.add(cite)
                patent['citation'] = list(set(patent['citation']) - delete)
                if len(patent['citation']) >= 10:
                    test_other.append(patent)
                    test_subset.add(patent['number'])
                    inventors |= set(patent['inventor'])
                    assignees |= set(patent['assignee'])
                    ipcrs |= set(patent['ipcr'])
                    train_subset |= set(patent['citation'])

        train_other = []
        with open(inputfile + 'train_patent_other.txt', 'r', encoding='utf-8') as tpo:
            for line in tqdm(tpo):
                patent = json.loads(line.strip())
                if patent['number'] in train_subset:
                    # chd.add_patent_is_train(patent['number'], True)
                    train_other.append(patent)
                    inventors |= set(patent['inventor'])
                    assignees |= set(patent['assignee'])
                    ipcrs |= set(patent['ipcr'])

        self.chd.add_inventors_num(len(inventors))
        self.chd.add_assignees_num(len(assignees))
        self.chd.add_ipcrs_num(len(ipcrs))
        self.chd.add_patents_all_num(len(test_subset | train_subset))
        self.chd.add_patents_test_num(len(test_subset))
        print(len(test_subset | train_subset))

        print("train_subset......")
        i = 0
        i = self._add_index(train_subset, i)
        print("inventors......")
        i = self._add_index(inventors, i)
        print("assignees......")
        i = self._add_index(assignees, i)
        print("ipcr......")
        i = self._add_index(ipcrs, i)
        print("test_subset......")
        i = self._add_index(test_subset, i)

        print("test_other......")
        for test in tqdm(test_other):
            index = self.chd.get_index(test['number'])
            self.chd.add_patent_is_train(index, False)
            self.chd.add_inventors(index, self._add_other(test['inventor']))
            self.chd.add_assignee(index, self._add_other(test['assignee']))
            self.chd.add_ipcr(index, self._add_other(test['ipcr']))
        print("train_other......")
        for train in tqdm(train_other):
            index = chd.get_index(train['number'])
            chd.add_patent_is_train(index, True)
            self.chd.add_inventors(index, self._add_other(train['inventor']))
            self.chd.add_assignee(index, self._add_other(train['assignee']))
            self.chd.add_ipcr(index, self._add_other(train['ipcr']))

        with open(inputfile + 'test_patent_content.txt', 'r', encoding='utf-8') as tpc:
            for line in tqdm(tpc):
                patent = json.loads(line.strip())
                # test_number = set(chd.all_target_patent())
                if patent['number'] in test_subset:
                    self._add_content(patent)

        with open(inputfile + 'train_patent_content.txt', 'r', encoding='utf-8') as trpc:
            for line in tqdm(trpc):
                patent = json.loads(line.strip())
                # train_number = set(chd.all_candidate_patent())
                try:
                    self._add_content(patent)
                except Exception as e:
                    # print(e)
                    continue
        self.chd.close()

    def _add_index(self, items, i):
        for num in tqdm(items):
            self.chd.add_index(num, i)
            self.chd.add_raw(i, num)
            i += 1
        return i

    def _add_other(self, items):
        index = []
        for inv in items:
            index.append(self.chd.get_index(inv))
        return index

    def _add_content(self, patent):
        index = self.chd.get_index(patent['number'])
        self.chd.add_title(index, patent['title'])
        self.chd.add_abstract(index, patent['abstract'])
        self.chd.add_full_text(index, patent['title'] + '.    ' + patent['abstract'] + '    ' + patent['claim'])

    def selfcitation(self, inputfile):
        sc = set()
        with open(inputfile + 'train_patent_other.txt', 'r', encoding='utf-8') as tpo:
            for line in tqdm(tpo):
                patent = json.loads(line.strip())
                for num in patent['citation']:
                    if num == patent['number']:
                        sc.add(num)
        print(sc)
class get_patent():
    def __init__(self):
        pass

    def GetTest(self):
        pass

    def GetTrain(self):
        pass


if __name__ == '__main__':
    patent_inf = {}
    inf_target = []
    process = cpu_count()
    target_lines = []
    GetData = get_data()
    # GetData.GetInf('../patent_inf/patent/uspto_patent2patent_all.json', '../patent_inf/patent/', test=True)
    # GetData.GetTrain('../patent_inf/patent/test_citation_number.txt', '../patent_inf/patent/')
    # GetData.GetInf('../patent_inf/patent/uspto_train_patent.json', '../patent_inf/patent/', test=False)
    GetData.Select('../patent_inf/patent/', chd=citation_h5py_dataset(filename="crdataset.h5py", dataset="aan", write='True', write_mode='w'))

    # GetData.selfcitation('../patent_inf/patent/')
    # chd = citation_h5py_dataset(filename="crdataset.h5py", dataset="aan", write='False', write_mode='r')
    # print(len(chd.all_target_patent()))
