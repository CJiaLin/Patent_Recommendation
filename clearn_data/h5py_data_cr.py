import h5py
from os.path import isfile
import numpy as np
import pickle as pkl

# 父类
class h5py_dataset(object):

    def __init__(self, filename, write=False, compression_level=8, write_mode="a", driver="core"):
        # compression_level 压缩等级
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
        for k, v in dict_data.items():
            self._file[group][key].attrs[k] = v

    def get_dict(self, group, key):
        group, key = self._check_keys(group, key)
        return {k: v for k, v in self._file[group][key].attrs.items()}

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

    def add_patents_all_num(self, patents_all_num):
        self[self.dataset + "_patents_all_num"] = patents_all_num

    def get_patents_all_num(self):
        return self[self.dataset + "_patents_all_num"]

    def add_patents_train_num(self, patents_train_num):
        self[self.dataset + "_patents_train_num"] = patents_train_num

    def get_patents_train_num(self):
        return self[self.dataset + "_patents_train_num"]

    def add_inventors_num(self, inventors_num):
        self[self.dataset + "_inventors_num"] = inventors_num

    def get_inventors_num(self):
        return self[self.dataset + "_inventors_num"]

    def add_assignees_num(self, assignees_num):
        self[self.dataset + "_assignees_num"] = assignees_num

    def get_assignees_num(self):
        return self[self.dataset + "_assignees_num"]

    def add_ipcrs_num(self, ipcrs_num):
        self[self.dataset + "_ipcrs_num"] = ipcrs_num

    def get_ipcrs_num(self):
        return self[self.dataset + "_ipcrs_num"]

    def add_patents_test_num(self, patents_test_num):
        self[self.dataset + "/patents_test_num"] = patents_test_num

    def get_patents_test_num(self):
        return self[self.dataset + "/patents_test_num"]

    def add_patents_train_num(self, patents_train_num):
        self[self.dataset + "/patents_train_num"] = patents_train_num

    def get_patents_train_num(self):
        return self[self.dataset + "/patents_train_num"]

    def add_patent_is_train(self, id, is_train):
        self.add_pickle(self.dataset + "/is_train", id, is_train)

    def get_patent_is_train(self, id):
        return self.get_pickle(self.dataset + "/is_train", id)

    def add_abstract(self, id, abstract):
        self.add_pickle(self.dataset + "/abstract", id, abstract)

    def get_abstract(self, id):
        return self.get_pickle(self.dataset + "/abstract", id)

    def add_abstract_fenci(self, id, abstract_fenci):
        self.add_pickle(self.dataset + "/abstract_fenci", id, abstract_fenci)

    def get_abstract_fenci(self, id):
        return self.get_pickle(self.dataset + "/abstract_fenci", id)

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

    def add_assignee(self, id, assignee):
        self.add_pickle(self.dataset + "/assignee", id, assignee)

    def get_assignee(self, id):
        return self.get_pickle(self.dataset + "/assignee", id)

    def add_inventors(self, id, inventors):
        self.add_pickle(self.dataset + "/inventors", id, inventors)

    def get_inventors(self, id):
        return list(map(str, self.get_pickle(self.dataset + "/inventors", id)))

    def add_ipcr(self, id, ipcr):
        self.add_pickle(self.dataset + "/ipcr", id, ipcr)

    def get_ipcr(self, id):
        return self.get_pickle(self.dataset + "/ipcr", id)

    def add_index(self, id, index):
        self.add_pickle(self.dataset + "/index", id, index)

    def get_index(self, id):
        return self.get_pickle(self.dataset + "/index", id)

    def add_raw(self, index, id):
        self.add_pickle(self.dataset + "/raw", index, id)

    def get_raw(self, index):
        return self.get_pickle(self.dataset + "/raw", index)

    def add_full_fenci(self, id, full_fenci):
        self.add_pickle(self.dataset + "/full_fenci", id, full_fenci)

    def get_full_fenci(self, id):
        return self.get_pickle(self.dataset + "/full_fenci", id)

    def add_full_text(self, id, full_text):
        self.add_pickle(self.dataset + "/full_text", id, full_text)

    def get_full_text(self, id):
        return self.get_pickle(self.dataset + "/full_text", id)

    def all_full_text(self):
        patents = [self.get_full_text(i) for i in range(self.get_patents_all_num())]
        start = self.get_patents_train_num()+self.get_inventors_num()+self.get_assignees_num()+self.get_ipcrs_num()
        patents += [self.get_full_text(i) for i in range(start, start + self.get_patents_test_num())]
        return patents

    def all_candidate_full_text(self):
        return [self.get_full_text(i) for i in range(self.get_patents_train_num())]

    def all_target_full_text(self):
        start = self.get_patents_train_num()+self.get_inventors_num()+self.get_assignees_num()+self.get_ipcrs_num()
        return [self.get_full_text(i) for i in range(start, start+self.get_patents_test_num())]

    def all_target_patent(self):
        all_tatget_patent = []
        start = self.get_patents_train_num()+self.get_inventors_num()+self.get_assignees_num()+self.get_ipcrs_num()
        for i in range(start, start+self.get_patents_test_num()):
            all_tatget_patent.append(i)
        return all_tatget_patent

    def all_candidate_patent(self):
        all_candidate_patent = []
        for i in range(self.get_patents_train_num()):
            # if self.get_patent_is_train(i):
            all_candidate_patent.append(i)

        return all_candidate_patent

    def all_patent(self):
        all_patent = [i for i in range(self.get_patents_train_num())]
        start = self.get_patents_train_num()+self.get_inventors_num()+self.get_assignees_num()+self.get_ipcrs_num()
        all_patent += [j for j in range(start, start+self.get_patents_test_num())]
        return all_patent


if __name__ == '__main__':
    chd = citation_h5py_dataset(filename="crdataset.h5py",dataset="aan")
    print(chd._file)
    # for i in chd.all_target_patent():
    #     print(type(i))
    cp = chd.all_candidate_patent()
    # for i in chd.all_candidate_patent():
    #     print(i)
