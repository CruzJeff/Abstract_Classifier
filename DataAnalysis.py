# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 01:19:19 2018

@author: User
"""


import pandas as pd
import string
from nltk.corpus import stopwords
import re
from collections import Counter
from collections import OrderedDict
from operator import itemgetter


eng_stopwords = set(stopwords.words("english"))

Chem_Abstracts = pd.read_csv("Chem_Train.csv", encoding="utf-8")

CS_Abstracts = pd.read_csv("CS_Train.csv", encoding="cp1252")
Eco_Abstracts = pd.read_csv("Eco_Train.csv", encoding="latin-1")
Bio_Abstracts = pd.read_csv("Bio_Train.csv", encoding="latin-1")

Chem_Test = pd.read_csv("Chem_Test.csv", encoding="utf-8")
CS_Test = pd.read_csv("CS_Test.csv", encoding="cp1252")
Eco_Test = pd.read_csv("Eco_Test.csv",encoding="latin-1")
Bio_Test = pd.read_csv("Bio_Test.csv", encoding='latin-1')


#Turn this into machine learning problem?
#Find more abstracts, and create classification model to see what abstracts get confused as others


def Analyze(dataframe):
    
    dataframe["Num_Words"] = dataframe["Abstracts"].apply(lambda x: len(str(x).split()))

    dataframe["num_unique_words"] = dataframe["Abstracts"].apply(lambda x: len(set(str(x).split())))

    dataframe["num_stopwords"] = dataframe["Abstracts"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

    dataframe["num_punctuations"] = dataframe["Abstracts"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

Analyze(Chem_Abstracts)
Analyze(CS_Abstracts)
Analyze(Eco_Abstracts)
Analyze(Bio_Abstracts)

words = []

for passage in Chem_Abstracts["Abstracts"]:
    words.append(re.findall(r'\w+', passage))

words = [j for i in words for j in i]

words=  [w for w in words if (not w.lower() in eng_stopwords and w[0] != '/' and w[0] != 'x' and len(w) != 1)]

cap_words = [word.upper() for word in words]

word_counts = Counter(cap_words)

sorted_counts = OrderedDict(sorted(word_counts.items(), key=itemgetter(1),reverse = True))

sorted_counts.keys()