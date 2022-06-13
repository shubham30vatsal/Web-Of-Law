import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.model_selection import train_test_split
from textacy.datasets.supreme_court import SupremeCourt
import numpy as np
import re
import unicodedata
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout, Input, BatchNormalization
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt
import itertools
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
#from transformers import *
from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import gc
import math
import json
import stanza
from tensorflow.keras import *
import tensorflow as tf
from tensorflow.keras import *
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import TFRobertaModel,RobertaTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import RandomUniform

from numpy.random import seed
import random as python_random
import os
import sys

np.random.seed(1)
python_random.seed(1)
tf.random.set_seed(1)


sc = SupremeCourt()
print(sc.info)
sc.download()

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

issue_codes = list(sc.issue_area_codes.keys()) # 15 labels
print(issue_codes)
issue_codes.sort()
issue_codes = [str(ic) for ic in issue_codes]

labels_index = dict(zip(issue_codes, np.arange(len(issue_codes))))
print(labels_index)
count=0
#summarizer = pipeline("summarization",model='facebook/bart-large-cnn',tokenizer='facebook/bart-large-cnn',framework='tf')
summarizer = pipeline("summarization")

textfile = open("summarized_usdb.txt", "w")
for record in sc.records():
        
        count=count+1
        print("Count")
        print(count)
        # if count==100:
        #   break
        
        if record[1]['issue'] == None: # some cases have None as an issue
            labels.append(labels_index['-1'])
        else:
            labels.append(labels_index[record[1]['issue'][:-4]])
        
        new_sen=record[0].split("Footnotes")[0]

        if len(new_sen.split())<=512:
          print("Length of Summarized Final Text")
          print(len(new_sen.split()))
          textfile.write("--- "+new_sen + "\n")

        else:
          n_splits=len(new_sen.split())/1024
          n_splits=math.floor(n_splits)
          if n_splits==0:
              n_splits=1
          print("Number Of Splits")
          print(n_splits)
          n_wordspersplit=512/n_splits
          print("Number Of Words Per Split")
          print(n_wordspersplit)
          net_sum=[]
          new_sen=new_sen.split()
          for split in range(n_splits):
              #print(len(new_sen[split*1025:(split+1)*1025]))
              # if len(new_sen[split*1025:(split+1)*1025])<=1024:
              #   summarized=new_sen[split*1025:len(new_sen)+1]
              #   print("Length of Summarized Text")
              #   print(len(new_sen[split*1025:(split+1)*1025]))
              #   net_sum.append(summarized)

              # else:
              #max=round(n_wordspersplit)
              #lenplus=len(new_sen.split())+1
              # if len(new_sen[split*1025:lenplus])<=n_wordspersplit:
              #   print("True")
              #   print(split*1025)
              #   print(lenplus)
              #   min=len(new_sen[split*1025:lenplus])
              #   max=len(new_sen[split*1025:lenplus])
              # else:
              #min=math.floor(n_wordspersplit)
              max=math.floor(n_wordspersplit)
              min=max-5
              print("Min")
              print(min)
              print("Max")
              print(max)
              #summarized=summarizer(new_sen[split*1025:(split+1)*1025], min_length=math.floor(n_wordspersplit), max_length=math.floor(n_wordspersplit+10))
              #print(len(new_sen[split*512:(split+1)*512]))
              #print(new_sen[split*512:(split+1)*512])
              temp=' '.join(new_sen[split*1024:(split+1)*1024])
              #print(temp)
              summarized=summarizer(temp, min_length=min, max_length=max, truncation=True)
              
              #summarized=summarizer(new_sen[split*1024:(split+1)*1024], min_length=min, max_length=max)
              #summarized=summarizer(new_sen[split*1025:(split+1)*1025], min_length=166, max_length=176)
              
              print("Length of Summarized Split")
              print(len(summarized[0]['summary_text'].split()))
              
              #print(summarized)
              #print(summarized[0]['summary_text'])
              net_sum.append(summarized[0]['summary_text'])
          #summarized = summarizer(new_sen, min_length=450, max_length=500)
          
          new_sen_summary=' '.join(net_sum)
          #print(new_sen)
          print("Length of Summarized Final Text")
          print(len(new_sen_summary.split()))
          textfile.write("--- "+new_sen_summary + "\n")
          textfile.flush()

textfile.close()
len_list = [len(ele.split()) for ele in texts]

print(labels)
print(len(labels))
res = 0 if len(len_list) == 0 else (float(sum(len_list)) / len(len_list))
print("Average Length %s" % res) 
print('Found %s texts.' % len(texts))
print('Found %s labels.' % len(labels_index))
    