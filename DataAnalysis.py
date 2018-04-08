
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
import matplotlib.pyplot as plt
import numpy as np

eng_stopwords = set(stopwords.words("english"))

Chem_Abstracts = pd.read_csv("Chem_Train.csv", encoding="utf-8")

CS_Abstracts = pd.read_csv("CS_Train.csv", encoding="cp1252")
Eco_Abstracts = pd.read_csv("Eco_Train.csv", encoding="latin-1")
Bio_Abstracts = pd.read_csv("Bio_Train.csv", encoding="latin-1")

Chem_Abstracts["Label"] = ["Chemistry" for x in range(len(Chem_Abstracts))]
CS_Abstracts["Label"] = ["Computer Science" for x in range(len(CS_Abstracts))]
Eco_Abstracts["Label"] = ["Ecology" for x in range(len(Eco_Abstracts))]
Bio_Abstracts["Label"] = ["Biology" for x in range(len(Bio_Abstracts))]

Chem_Test = pd.read_csv("Chem_Test.csv", encoding="utf-8")
CS_Test = pd.read_csv("CS_Test.csv", encoding="cp1252")
Eco_Test = pd.read_csv("Eco_Test.csv",encoding="latin-1")
Bio_Test = pd.read_csv("Bio_Test.csv", encoding='latin-1')

Chem_Test["Label"] = ["Chemistry" for x in range(len(Chem_Test))]
CS_Test["Label"] = ["Computer Science" for x in range(len(CS_Test))]
Eco_Test["Label"] = ["Ecology" for x in range(len(Eco_Test))]
Bio_Test["Label"] = ["Biology" for x in range(len(Bio_Test))]



def Analyze(dataframe):
    
    dataframe["Num_Words"] = dataframe["Abstracts"].apply(lambda x: len(str(x).split()))

    dataframe["num_unique_words"] = dataframe["Abstracts"].apply(lambda x: len(set(str(x).split())))

    dataframe["num_stopwords"] = dataframe["Abstracts"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

    dataframe["num_punctuations"] = dataframe["Abstracts"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

Analyze(Chem_Abstracts)
Analyze(CS_Abstracts)
Analyze(Eco_Abstracts)
Analyze(Bio_Abstracts)
Analyze(Chem_Test)
Analyze(Bio_Test)
Analyze(CS_Test)
Analyze(Eco_Test)

Train = pd.concat([Eco_Abstracts,CS_Abstracts,Bio_Abstracts, Chem_Abstracts], ignore_index=True)

Test =  pd.concat([Eco_Test,CS_Test,Bio_Test, Chem_Test], ignore_index=True)

#Random Forest Classifier

import xgboost

X_Train = Train.iloc[:,2:6]
Y_Train = Train["Label"]

X_Test = Test.iloc[:,2:6]
Y_Test = Test["Label"]


words = []
for passage in Train["Abstracts"]:
    words.append(re.findall(r'\w+', passage))

words = [j for i in words for j in i]

words=  [w for w in words if (not w.lower() in eng_stopwords and w[0] != '/' and w[0] != 'x' and len(w) != 1)]

cap_words = [word.upper() for word in words]

word_counts = Counter(cap_words)

sorted_counts = OrderedDict(sorted(word_counts.items(), key=itemgetter(1),reverse = True))

top_100 = list(sorted_counts)[0:100]

#Create bag of words
from keras.preprocessing.text import Tokenizer
max_words = 100 
tokenize = Tokenizer(num_words = max_words, char_level = False)
tokenize.fit_on_texts(words)
x_train = pd.DataFrame(tokenize.texts_to_matrix(Train["Abstracts"]))
x_test = pd.DataFrame(tokenize.texts_to_matrix(Test["Abstracts"]))
X_Train = pd.concat([X_Train,x_train], axis = 1)
X_Test = pd.concat([X_Test,x_test], axis = 1)

boost = xgboost.XGBClassifier(n_estimators=800)
boost.fit(X_Train,Y_Train)

pred = list(boost.predict(X_Test))
true = list(Y_Test)

error = 0

for x in range(len(true)):
    if true[x] != pred[x]:
        error += 1

y_actu = pd.Series(true, name='Actual')
y_pred = pd.Series(pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)
    
#Turn this into machine learning problem?
#Find more abstracts, and create classification model to see what abstracts get confused as others
