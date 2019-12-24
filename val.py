# -*- coding:utf-8 -*-
import bilsm_crf_model
import process_data
import numpy as np
import pandas as pd
import re
import string
from zhon.hanzi import punctuation
import matplotlib.pyplot as plt

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
model.load_weights('model/crf.h5')

def getRawResult(predict_text):
    str, length = process_data.process_data(predict_text, vocab)
    raw = model.predict(str)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]
def getPersonName(predict_text):
    str, length = process_data.process_data(predict_text, vocab)
    raw = model.predict(str)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]
    per, loc, org = '', '', ''
    for s, t in zip(predict_text, result_tags):
        if t in ('B-PER', 'I-PER'):
            per += ' ' + s if (t == 'B-PER') else s
        if t in ('B-ORG', 'I-ORG'):
            org += ' ' + s if (t == 'B-ORG') else s
        if t in ('B-LOC', 'I-LOC'):
            loc += ' ' + s if (t == 'B-LOC') else s
    return per


def getOrgName(predict_text):
    str, length = process_data.process_data(predict_text, vocab)
    raw = model.predict(str)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]
    per, loc, org = '', '', ''
    for s, t in zip(predict_text, result_tags):
        if t in ('B-PER', 'I-PER'):
            per += ' ' + s if (t == 'B-PER') else s
        if t in ('B-ORG', 'I-ORG'):
            org += ' ' + s if (t == 'B-ORG') else s
        if t in ('B-LOC', 'I-LOC'):
            loc += ' ' + s if (t == 'B-LOC') else s
    return org

index=0
for text in df_train["text"]:
    text=clean_data(text)
    # print (text)
    # print ("------------------------------")
    sentences=text.split('。')
    res=[]
    for sen in sentences:
        # print (sen)
        # print("-----------")
        name=getPersonName(sen)
        if( len(name)!=0 or name!="" ):
            name = name.split(' ')
            name = list(filter(None,name))
            res=res+name
    res=list(set(res))
    df_name["name"][index]=";".join(res)
    index=index+1
    # print (res)