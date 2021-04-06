'''
    coding: utf-8
    author: Tejul Pandit
'''
##IMPORT LIBRARIES

import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import os
import pandas as pd

##DEFINING FUNCTIONS FOR DE-DUPLICATION
#function to substitute specific characters with relevant information
def clean_headlines(text, remove_stopwords=False):
    text = text.lower() 
    text = text.translate(str.maketrans('', '', string.punctuation)) #replace punctuations
    text = re.sub(r'[0-9]', '', text) #remove numbers
    re.sub(' +', ' ', text)

    # Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in text.split() if not w in stops]

        text = " ".join(words)

    return text

#function to cluster similar news articles
def clustering(model, clean_test_headlines, threshold_value):
    final_score = [[] for i in range(len(clean_test_headlines))]
    max_score = [[] for i in range(len(final_score))]
    cluster_pairs = []
    num_list = list(range(0, len(max_score))) 
    
    for i in range(len(clean_test_headlines)):
        for j in range(len(clean_test_headlines)):
            if i != j:
                try:
                    score = model.n_similarity(clean_test_headlines[i].split(), clean_test_headlines[j].split())
                    final_score[i].append(score) 
                except Exception as e:
                    final_score[i].append(e)
            else:
                final_score[i].append(0.0) 
                
    
    for i in range(len(final_score)):
        (value, index) = max((x, (j)) for j, x in enumerate(final_score[i]))
        max_score[i].append((value, index))
        
    for i in range(int(len(max_score))):
        if (max_score[i][0][0] >= threshold_value) & (i == max_score[max_score[i][0][1]][0][1]):
            cluster_pairs.append((i, max_score[i][0][1]))
            
    for i in range(len(cluster_pairs)):
        num_list[cluster_pairs[i][1]] = num_list[cluster_pairs[i][0]]
    
    return num_list