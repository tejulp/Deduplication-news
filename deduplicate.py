'''
    coding: utf-8
    author: Tejul Pandit
'''
##IMPORT LIBRARIES

import os
import collections
from nltk.stem import SnowballStemmer
import gensim 
from gensim import utils
from gensim.models.deprecated.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import def_func

#Code for de-duplication
def deduplication(df):
    df['clean_headlines'] = df['title'].apply(def_func.clean_rawData)
    lst_headlines=[]
    
    for i in range(len(df)):
        lst_headlines.append(TaggedDocument(df['clean_headlines'][i].split(), df[df.index == i]['index']))

    model = Doc2Vec(dm = 1, min_count=1, window=10, size=150, sample=1e-4, negative=10)
    model.build_vocab(lst_headlines)

    # Train the model with 20 epochs 
    for epoch in range(20):
        model.train(labeled_questions,epochs=model.iter,total_examples=model.corpus_count)
        print("Epoch #{} is complete.".format(epoch+1))
    
    return model
