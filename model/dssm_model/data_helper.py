#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:data_helper.py
# @Author: Michael.liu
# @Date:2020/8/25 15:27
# @Desc: this code is ....
import numpy as np
from pyhanlp import *
from model.dssm_model.config import Config

conf= Config()


def pyHanlpSeg(content):
    seg_list = HanLP.segment("".join(content))
    hanlpSegList = []
    for item in seg_list:
        hanlpSegList.append(item.word)
        rec = ' '.join(hanlpSegList)
    return rec


def gen_word_set(filepath,outPath="./words.txt"):
    word_set = set()
    with open(filepath, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 3:
                continue
            query,doc,label = spline
            if label== '0':
                continue
            query_seg = list(query)
            doc_seg = list(doc)
            for w in query_seg:
                word_set.add(w)
            for each in doc_seg:
                for w in each:
                    word_set.add(w)
    with open(outPath, 'w',encoding='utf8') as o:
        for w in word_set:
            o.write(w + '\n')
    pass

def convert_word2id(query,vocab_map):
    ids = []
    query_list = list(query)
    for w in query_list :
        if w in vocab_map:
            ids.append(vocab_map[w])
        else:
            ids.append(conf.unk)
            #ids.append(vocab_map[conf.vocab_map])
    while len(ids) < conf.max_seq_len:
        ids.append(vocab_map[conf.pad])
    return ids[:conf.max_seq_len]



def get_data(file):
    data_map = {'query': [], 'query_len': [], 'doc_pos': [], 'doc_pos_len': [] }
    i=1
    with open(file, 'r',encoding='utf8') as f:
        for line in f.readlines():
            i+=1
            spline = line.strip().split("\t")
            if len(spline) < 3:
                continue
            query, doc, label = spline
            if label == "0":
                continue
            #query_arr,query_len = [],[]
            #query_seg = pyHanlpSeg(query)
            #doc_seg = pyHanlpSeg(doc)
            data_map['query'].append(convert_word2id(query,conf.vocab_map))
            data_map['query_len'].append(len(query)  if len(query) < conf.max_seq_len else conf.max_seq_len)
            data_map['doc_pos'].append(convert_word2id(doc,conf.vocab_map))
            data_map['doc_pos_len'].append(len(doc) if len(doc) < conf.max_seq_len else conf.max_seq_len)
            #data_map['doc_neg'].extend()
            pass
            print(i)
            print(data_map)
    return data_map

if __name__ == '__main__':
    #pyHanlpSeg("我爱北京天安门")
    get_data('./train.txt')