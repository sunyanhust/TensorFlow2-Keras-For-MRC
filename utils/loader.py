import os
import json
import numpy as np
import pandas as pd
import pickle as pk


def dump_json(obj, fpath, encoding='UTF-8', is_ascii=False, indent=4):
    with open(fpath, 'w', encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=is_ascii, indent=indent)


def load_json(fpath, encoding='UTF-8'):
    with open(fpath, 'r', encoding=encoding) as f:
        return json.load(f)


def dump_pkl(obj, fpath):
    with open(fpath, 'wb') as f:
        pk.dump(obj, f)


def load_pkl(fpath):
    with open(fpath, 'rb') as f:
        return pk.load(f)


def dump_npy(obj, fpath):
    np.save(fpath, obj)


def load_npy(fpath):
    return np.load(fpath)


def dump_csv(obj, fpath, header=False, index=False):
    pd.DataFrame(obj).to_csv(fpath, header=header, index=index)


def load_csv(fpath, header=None):
    pd.read_csv(fpath, header=header)


def load(fpath, **kwargs):
    load_dict = {
        '.json': load_json,
        '.pkl': load_pkl,
        '.npy': load_npy,
        '.csv': load_csv
    }
    _, ext = os.path.splitext(fpath)
    return load_dict[ext](fpath)


def dump(obj, fpath, **kwargs):
    dump_dict = {
        '.json': dump_json,
        '.pkl': dump_pkl,
        '.npy': dump_npy,
        '.csv': dump_csv
    }
    _, ext = os.path.splitext(fpath)
    return dump_dict[ext](obj, fpath, **kwargs)


def mkdir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
