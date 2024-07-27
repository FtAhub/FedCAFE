import h5py
import scipy.io as scio
import os
import numpy as np

def load_data(path):
    if 'FLICKR-25K' in path:
        file = h5py.File(path)
        images = file['images'][:]
        labels = file['LAll'][:]
        tags = file['YAll'][:]
        file.close()

    return images, tags, labels


def load_pretrain_model(path):
    return scio.loadmat(path)

def split_data(images, tags, labels, opt):
    X = {}
    X['query'] = images[0: opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]

    L = {}
    L['query'] = labels[0: opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L