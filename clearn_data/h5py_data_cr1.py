import h5py
from os.path import isfile
import numpy as np
import _pickle as pkl

# 父类
class h5py_dataset(object):

    def __init__(self,filename,write=False,compression_level=8,write_mode="x",driver="core"):
        self._keys = None
        self._comp_level = compression_level
        self._meta_data = {}

        if not write:
            if not isfile(filename):
                raise Exception("File %s does not exist" %filename)     # 引发一个异常
            self._file = h5py.File(filename,mode="r",libver="latest",driver=driver)
        else:
            if write_mode in ["x","w-"] and isfile(filename):
                raise Exception("File %s already; either change write_mode or filename" %filename)
            self._file = h5py.File(filename,mode=write_mode,libver="latest",driver=driver)

    def close(self):
        self._file.close()

    def keys(self):
        if self._keys:
            return self._keys
        self._keys = self._file.keys()
        return [key for key in self._keys]

    def _trans_key_str(self,key):
        if isinstance(key,(list,dict,set,tuple,type)):
            raise Exception("Error key type %s in h5py file" %str(type(key)))
        if isinstance(key,str):
            return key
        return str(key)

    def _check_keys(self,key1,key2):
        return self._trans_key_str(key1),self._trans_key_str(key2)

    def _check_dataset(self,group):
        self._require_object_dataset(group)

    def _require_object_dataset(self,group):
        self._file.require_dataset(group,shape=(1,),shuffle=True,dtype=np.int8,
                                   compression="gzip",compression_opts=self._comp_level)

    def add_pickle(self,group,key,obj):
        group,key = self._check_keys(group,key)
        self._check_dataset(group)
        self._file[group].attrs[key] = np.void(pkl.dumps(obj))

    def get_pickle(self, group, key):
        group, key = self._check_keys(group, key)
        return pkl.loads(self._file[group].attrs[key].tostring())

    def add_dict(self, group, key, dict_data):
        group, key = self._check_keys(group, key)
        self._check_dataset(group)
        for k, v in dict_data.iteritems():
            self._file[group][key].attrs[k] = v

    def get_dict(self, group, key):
        group, key = self._check_keys(group, key)
        return {k: v for k, v in self._file[group][key].attrs.iteritems()}

    def add_array(self, key, data):
        key = self._trans_key_str(key)
        self._file.create_dataset(key, data=data,compression='gzip', compression_opts=self._comp_level)

    def get_array(self, key):
        key = self._trans_key_str(key)
        return self._file[key].value

    def add_string(self, group, key, string):
        group, key = self._check_keys(group, key)
        self._check_dataset(group)
        self._file[group].attrs[key] = np.void(b"%s" % string)

    def get_string(self, group, key):
        group, key = self._check_keys(group, key)
        return self._file[group].attrs[key].tostring()

    def add_protobuf(self, group, key, protobuf):
        self.add_string(group, key, protobuf.SerializeToString())

    def get_protobuf(self, group, key, pbuff_cls):
        return pbuff_cls.FromString(self.get_string(self.get_string(group, key)))

    # 将类变为字典形式使用
    def __getitem__(self, key):
        return self._file[key].value

    def __setitem__(self, key, value):
        self._file[key] = value


# 子类
class citation_h5py_dataset(h5py_dataset):

    def __init__(self, filename="crdataset.h5py", dataset="aan", *args, **kwargs):
        self.filename = filename
        self.dataset = dataset
        super().__init__(self.filename, *args, **kwargs)

    def add_paper_all_num(self, papers_all_num):
        self[self.dataset + "_papers_all_num"] = papers_all_num

    def get_paper_all_num(self):
        return self[self.dataset + "_papers_all_num"]

    def add_authors_num(self, authors_num):
        self[self.dataset + "_authors_num"] = authors_num

    def get_authors_num(self):
        return self[self.dataset + "_authors_num"]

    def add_venues_num(self, venues_num):
        self[self.dataset + "_venues_num"] = venues_num

    def get_venues_num(self):
        return self[self.dataset + "_venues_num"]

    def add_papers_test_num(self, papers_test_num):
        self[self.dataset + "/papers_test_num"] = papers_test_num

    def get_papers_test_num(self):
        return self[self.dataset + "/papers_test_num"]

    def add_year(self, id, year):
        self.add_pickle(self.dataset + "/year",id,year)

    def get_year(self,id):
        return self.get_pickle(self.dataset + "/year",id)

    def add_paper_is_train(self, id, is_train):
        self.add_pickle(self.dataset + "/is_train", id, is_train)

    def get_paper_is_train(self, id):
        return self.get_pickle(self.dataset + "/is_train", id)

    def add_abstract(self, id, abstract):
        self.add_pickle(self.dataset + "/abstract", id, abstract)

    def get_abstract(self, id):
        return self.get_pickle(self.dataset + "/abstract", id)

    def add_citations(self, id, citations):
        self.add_pickle(self.dataset + "/citations", id, citations)

    def get_citations(self, id):
        return list(map(str, self.get_pickle(self.dataset + "/citations", id)))

    def add_outcite_num(self, id, outcite_num):
        self.add_pickle(self.dataset + "/outcite_num", id, outcite_num)

    def get_outcite_num(self, id):
        return self.get_pickle(self.dataset + "/outcite_num", id)

    def add_title(self, id, title):
        self.add_pickle(self.dataset + "/title", id, title)

    def get_title(self, id):
        return self.get_pickle(self.dataset + "/title", id)

    def add_title_raw(self, id, title_raw):
        self.add_pickle(self.dataset + "/title_raw", id, title_raw)

    def get_title_raw(self, id):
        return self.get_pickle(self.dataset + "/title_raw", id)

    def add_venue(self, id, venue):
        self.add_pickle(self.dataset + "/venue", id, venue)

    def get_venue(self, id):
        return self.get_pickle(self.dataset + "/venue", id)

    def add_authors(self, id, authors):
        self.add_pickle(self.dataset + "/authors", id, authors)

    def get_authors(self, id):
        return list(map(str, self.get_pickle(self.dataset + "/authors", id)))

    def add_authors_raw(self, id, authors_raw):
        self.add_pickle(self.dataset + "/authors_raw", id, authors_raw)

    def get_authors_raw(self, id):
        return self.get_pickle(self.dataset + "/authors_raw", id)

    def add_index(self, id, index):
        self.add_pickle(self.dataset + "/index", id, index)

    def get_index(self, id):
        return self.get_pickle(self.dataset + "/index", id)

    def add_full_text(self, id, full_text):
        self.add_pickle(self.dataset + "/full_text", id, full_text)

    def get_full_text(self, id):
        return self.get_pickle(self.dataset + "/full_text", id)

    def add_raw_full_text(self, id, raw_full_text):
        self.add_pickle(self.dataset + "/raw_full_text", id, raw_full_text)

    def get_raw_full_text(self, id):
        return self.get_pickle(self.dataset + "/raw_full_text", id)

    def all_full_text(self):
        return [self.get_full_text(i) for i in range(self.get_paper_all_num())]

    def all_target_paper(self, t="str"):
        all_tatget_paper = []
        for i in range(self.get_paper_all_num()):
            if not self.get_paper_is_train(i):
                all_tatget_paper.append(i)
        if t == "int":
            return all_tatget_paper
        return list(map(int, all_tatget_paper))

    def all_candidate_paper(self, t="str"):
        all_candidate_paper = []
        for i in range(self.get_paper_all_num()):
            if self.get_paper_is_train(i):
                all_candidate_paper.append(i)
        if t == "int":
            return all_candidate_paper
        return list(map(int, all_candidate_paper))

    def all_paper(self, t="str"):
        all_paper = [i for i in range(self.get_paper_all_num())]
        if t == "int":
            return all_paper
        return list(map(int,all_paper))


if __name__ == '__main__':
    chd = citation_h5py_dataset(filename="crdataset.h5py",dataset="aan")
    print(chd._file)
    # for i in chd.all_target_paper():
    #     print(type(i))
    cp = chd.all_candidate_paper()
    # for i in chd.all_candidate_paper():
    #     print(i)
