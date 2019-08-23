import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Manager
import json
from collections import defaultdict

def writefile(file_name,content):
    with open('../patent_inf/' + file_name, 'w') as file:
        for key,value in tqdm(content.items()):
            file.write(json.dumps({'patent_number': key, 'content': value}) + '\n')


def readfile(file_name):
    with open('../patent_inf/USPTO/' + file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def GetNetwork(test_number):
    assignee2id = {}
    i = 0
    with open('../patent_inf/assignee.txt', 'r') as ass:
        for line in ass:
            num = line.strip()
            assignee2id[num] = i
            i += 1

    with open('../patent_inf/train_patent_inf_new.txt', 'r', encoding='utf-8') as file:
        with open('../patent_inf/network/train_citation.txt', 'w', encoding='utf-8') as pc:
            with open('../patent_inf/network/train_inventor.txt', 'w', encoding='utf-8') as pi:
                with open('../patent_inf/network/train_assignee.txt', 'w', encoding='utf-8') as pa:
                    with open('../patent_inf/network/train_assignee.txt', 'w', encoding='utf-8') as ass:
                        for line in tqdm(file):
                            patent = json.loads(line)
                            patent_id = patent['number']
                            citations = patent['citation']
                            inventors = patent['inventor']
                            assignees = patent['assignee']
                            for inventor in inventors:
                                id = inventor['id']
                                pi.write('p' + patent_id + '\t' + 'i' + id + '\n')

                            for assignee in assignees:
                                id = assignee['id']
                                try:
                                    pa.write('p' + patent_id + '\t' + 'a' + str(assignee2id[id]) + '\n')
                                except KeyError:
                                    ass.write(id + '\n')
                                    assignee2id[id] = i
                                    pa.write('p' + patent_id + '\t' + 'a' + str(i) + '\n')
                                    i += 1

                            for citation in citations:
                                try:
                                    id = citation['citation_id']
                                    if id.isdigit() and id in test_number:
                                        pc.write('p' + patent_id + '\t' + 'p' + id + '\n')
                                except KeyError:
                                    print(patent_id)
                                except Exception as e:
                                    print(patent_id)
                                    print(e)


def GetSingleInf():
    inventor_num = set()
    assignee_num = set()

    with open('../patent_inf/train_patent_inf.txt', 'r', encoding='utf-8') as file:
        with open('../patent_inf/inventor.txt', 'w', encoding='utf-8') as pi:
            with open('../patent_inf/assignee.txt', 'w', encoding='utf-8') as pa:
                for line in tqdm(file):
                    patent = json.loads(line)
                    inventors = patent['inventor']
                    assignees = patent['assignee']

                    for inventor in inventors:
                        id = inventor['id']
                        inventor_num.add(id)

                    for assignee in assignees:
                        id = assignee['id']
                        assignee_num.add(id)

                for inventor in inventor_num:
                    pi.write(inventor + '\n')

                for assignee in assignee_num:
                    pa.write(assignee + '\n')


def GetText():
    with open('../patent_inf/test_patent_inf_new.txt', 'r', encoding='utf-8') as file:
        with open('../patent_inf/network/test_content.txt', 'w', encoding='utf-8') as pc:
            for line in tqdm(file):
                try:
                    patent = json.loads(line)
                    patent_id = patent['number']
                    pc.write(json.dumps({'patent_number': patent_id, 'title': patent['title'], 'abstract': patent['abstract'], 'claim': patent['claim']}) + '\n')
                except KeyError as k:
                    if 'abstract' not in patent.keys():
                        abstract = ''
                    else:
                        abstract = patent['abstract']

                    if 'title' not in patent.keys():
                        title = ''
                    else:
                        title = patent['title']

                    if 'claim' not in patent.keys():
                        claim = ''
                    else:
                        claim = patent['claim']

                    pc.write(json.dumps({'patent_number': patent_id, 'title': title, 'abstract': abstract, 'claim': claim}) + '\n')
                    # print(patent_id)
                    # print(k)
                except Exception as e:
                    print(patent_id)
                    print(e)


def GetBasic():
    number = set()
    with open('../patent_inf/test_patent_inf_new.txt', 'r', encoding='utf-8') as file:
        with open('../patent_inf/test_patent_inf_1.txt', 'w', encoding='utf-8') as tpi:
            with open('../patent_inf/network/test_number.txt', 'w', encoding='utf-8') as tn:
                for line in tqdm(file):
                    patent = json.loads(line)
                    if len(patent['inventor']) != 0 and len(patent['assignee']) != 0 and len(patent['citation']) >= 5 \
                            and patent['number'] not in number:
                        number.add(patent['number'])
                        tpi.write(json.dumps(patent) + '\n')
                        tn.write(patent['number'] + '\n')


def GetTestTrain():
    with open('../patent_inf/patent_inf.txt','r', encoding='utf-8') as pi:
        with open('../patent_inf/test_patent_inf_new.txt', 'w', encoding='utf-8') as testpi:
            with open('../patent_inf/train_patent_inf.txt', 'w', encoding='utf-8') as trainpi:
                for line in tqdm(pi):
                    patent = json.loads(line)
                    num = int(patent['number'])
                    if num >= 9854721:
                        testpi.write(json.dumps(patent) + '\n')
                    else:
                        trainpi.write(json.dumps(patent) + '\n')


def CreatID():
    i = 0
    with open('../patent_inf/network/train_number.txt', 'r') as tn:
        with open('../patent_inf/network/train_number2id.txt', 'w') as tn2id:
            for line in tn:
                num = line.strip()
                tn2id.write(num + '\t' + str(i) + '\n')
                i += 1

    with open('../patent_inf/network/assignee.txt', 'r') as ass:
        with open('../patent_inf/network/assignee2id.txt', 'w') as ass2id:
            for line in ass:
                num = line.strip()
                ass2id.write(num + '\t' + str(i) + '\n')
                i += 1

    with open('../patent_inf/network/inventor.txt', 'r') as inv:
        with open('../patent_inf/network/inventor2id.txt', 'w') as inv2id:
            for line in inv:
                num = line.strip()
                inv2id.write(num + '\t' + str(i) + '\n')
                i += 1


def BuildHIN():
    ass2id = {}
    with open('../patent_inf/network/assignee2id.txt', 'r') as ass:
        for line in ass:
            num = line.strip().split('\t')
            ass2id[num[1]] = num[0]

    inv2id = {}
    with open('../patent_inf/network/inventor2id.txt', 'r') as inv:
        for line in inv:
            num = line.strip().split('\t')
            inv2id[num[1]] = num[0]

    with open('../patent_inf/PatentHIN.txt', 'w') as ph:
        with open('../patent_inf/network/train_citation.txt', 'r') as pc:
            for line in tqdm(pc):
                ph.write(line)

        with open('../patent_inf/network/train_inventor.txt', 'r') as pi:
            for line in tqdm(pi):
                ph.write(line)

        with open('../patent_inf/network/inventor_inventor.txt', 'r') as ii:
            with open('../patent_inf/network/tra_inv_inv.txt', 'w') as tii:
                for line in tqdm(ii):
                    num = line.strip().split('\t')
                    ph.write('i' + inv2id[num[0][1:]] + '\t' + 'i' + inv2id[num[1][1:]] + '\n')
                    tii.write(inv2id[num[0][1:]] + '\t' + inv2id[num[1][1:]] + '\n')

        with open('../patent_inf/network/train_assignee.txt', 'r') as ta:
            for line in tqdm(ta):
                ph.write(line)

        with open('../patent_inf/network/assignee_assignee.txt', 'r') as aa:
            with open('../patent_inf/network/tra_ass_ass.txt', 'w') as taa:
                for line in tqdm(aa):
                    num = line.strip().split('\t')
                    ph.write('a' + ass2id[num[0][1:]] + '\t' + 'a' + ass2id[num[1][1:]] + '\n')
                    taa.write(ass2id[num[0][1:]] + '\t' + ass2id[num[1][1:]] + '\n')

    # with open('../patent_inf/network/patent_inventor.txt', 'r') as pi:
    #     with open('../patent_inf/network/train_inventor.txt', 'w') as ti:
    #         with open('../patent_inf/network/inventor2id.txt', 'w') as ii:
    #             inventor = set()
    #             inv2id = {}
    #             i = 1382839
    #             lines = []
    #             for line in tqdm(pi):
    #                 num = line.strip().split('\t')
    #                 lines.append(num)
    #                 if num[1][1:] not in inventor:
    #                     inventor.add(num[1][1:])
    #                     inv2id[num[1][1:]] = str(i)
    #                     ii.write(num[1][1:] + '\t' + str(i) + '\n')
    #                     i += 1
    #
    #             for line in tqdm(lines):
    #                 ti.write('p' + num2id[line[0][1:]] + '\t' + 'i' + inv2id[line[1][1:]] + '\n')
    #
    #
    # with open('../patent_inf/network/patent_assignee.txt', 'r') as pa:
    #     with open('../patent_inf/network/train_assignee.txt', 'w') as ta:
    #         with open('../patent_inf/network/assignee2id.txt', 'w') as ai:
    #             assignee = set()
    #             ass2id = {}
    #             lines = []
    #             for line in tqdm(pa):
    #                 num = line.strip().split('\t')
    #                 lines.append(num)
    #                 if num[1][1:] not in assignee:
    #                     assignee.add(num[1][1:])
    #                     ass2id[num[1][1:]] = str(i)
    #                     ai.write(num[1][1:] + '\t' + str(i) + '\n')
    #                     i += 1
    #
    #             for line in tqdm(lines):
    #                 ta.write('p' + num2id[line[0][1:]] + '\t' + 'a' + ass2id[line[1][1:]] + '\n')


if __name__ == '__main__':
    patent_inf = []
    inf_target = defaultdict(list)
    process = 32
    assignee_num = set()
    inventor_num = set()
    patent_assignee = []
    train_number = set()
    ex_number = set()

    # GetCitation(test_patent)
    # with open('../patent_inf/test_patent_inf_new.txt', 'r') as tni:
    #     with open('../patent_inf/network/test_number.txt', 'w') as tn:
    #         for line in tqdm(tni):
    #             patent = json.loads(line)
    #             num = patent['number']
    #             tn.write(num + '\n')
    #
    # with open('../patent_inf/network/train_number2id.txt', 'r') as tn:
    #     for line in tn:
    #         patent = line.strip().split('\t')
    #         train_number.add(patent[0])
    #
    # with open('../patent_inf/network/test_number.txt', 'r') as tn:
    #     for line in tn:
    #         patent = line.strip()
    #         train_number.add(patent)
    # with open('../patent_inf/inventor.txt', 'w', encoding='utf-8') as ass:
    #     for item in inventor_num:
    #         ass.write(item + '\n')
    # GetNetwork(train_number)
    # GetText()
    # GetBasic()

    BuildHIN()

    # assignee2id = {}
    # i = 0
    # for ass in assignee:
    #     assignee2id[ass] = i
    #     i += 1




