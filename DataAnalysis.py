
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 01:19:19 2018

@author: User
"""


#Importing the Libraries
import pandas as pd
import string
from nltk.corpus import stopwords
import re
from collections import Counter
from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import xgboost
from keras.preprocessing.text import Tokenizer

#English Stopwords (such as 'a', 'the', etc)
eng_stopwords = set(stopwords.words("english"))

#Loading in the Abstracts used as Training Data
Chem_Abstracts = pd.read_csv("Chem_Train.csv", encoding="utf-8")
CS_Abstracts = pd.read_csv("CS_Train.csv", encoding="cp1252")
Eco_Abstracts = pd.read_csv("Eco_Train.csv", encoding="latin-1")
Bio_Abstracts = pd.read_csv("Bio_Train.csv", encoding="latin-1")

#Creating the labels for the Training Data
Chem_Abstracts["Label"] = ["Chemistry" for x in range(len(Chem_Abstracts))]
CS_Abstracts["Label"] = ["Computer Science" for x in range(len(CS_Abstracts))]
Eco_Abstracts["Label"] = ["Ecology" for x in range(len(Eco_Abstracts))]
Bio_Abstracts["Label"] = ["Biology" for x in range(len(Bio_Abstracts))]

#Loading in the Abstracts used as Testing Data
Chem_Test = pd.read_csv("Chem_Test.csv", encoding="utf-8")
CS_Test = pd.read_csv("CS_Test.csv", encoding="cp1252")
Eco_Test = pd.read_csv("Eco_Test.csv",encoding="latin-1")
Bio_Test = pd.read_csv("Bio_Test.csv", encoding='latin-1')

#Creating the labels for the Testing Data
Chem_Test["Label"] = ["Chemistry" for x in range(len(Chem_Test))]
CS_Test["Label"] = ["Computer Science" for x in range(len(CS_Test))]
Eco_Test["Label"] = ["Ecology" for x in range(len(Eco_Test))]
Bio_Test["Label"] = ["Biology" for x in range(len(Bio_Test))]

#Plot Distribution of Disciplines in Training Set
labels = ["Chemistry", "Computer Science", "Ecology", "Biology"]
from IPython.core.pylabtools import figsize
figsize(12.5,3.5)
y_pos = np.arange(4)
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.figure(figsize=(2,10))
plt.bar(y_pos, [len(Chem_Abstracts), len(CS_Abstracts), len(Eco_Abstracts), len(Bio_Abstracts)])
plt.xticks(y_pos, labels)
plt.legend(fontsize=6)
plt.title('Number of Abstracts per Discipline (Training Data)', fontsize=24)
plt.show()

#Plot Distribution of Disciplines in Training Set
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.figure(figsize=(2,10))
plt.bar(y_pos, [len(Chem_Test), len(CS_Test), len(Eco_Test), len(Bio_Test)])
plt.xticks(y_pos, labels)
plt.legend(fontsize=6)
plt.title('Number of Abstracts per Discipline (Testing Data)', fontsize=24)
plt.show()

#Creating the Analyze method to calculate features of the Abstracts
def Analyze(dataframe):
    dataframe["Num_Words"] = dataframe["Abstracts"].apply(lambda x: len(str(x).split()))
    dataframe["num_unique_words"] = dataframe["Abstracts"].apply(lambda x: len(set(str(x).split())))
    dataframe["num_stopwords"] = dataframe["Abstracts"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    dataframe["num_punctuations"] = dataframe["Abstracts"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

#Doing the Analysis on all of the data
Analyze(Chem_Abstracts)
Analyze(CS_Abstracts)
Analyze(Eco_Abstracts)
Analyze(Bio_Abstracts)
Analyze(Chem_Test)
Analyze(Bio_Test)
Analyze(CS_Test)
Analyze(Eco_Test)

#Plotting the metrics

Num_Words_Mean = [np.mean(Chem_Abstracts["Num_Words"]),np.mean(CS_Abstracts["Num_Words"]),
                np.mean(Eco_Abstracts["Num_Words"]),np.mean(Bio_Abstracts["Num_Words"])]

Num_Unique_Words_Mean = [np.mean(Chem_Abstracts["num_unique_words"]),np.mean(CS_Abstracts["num_unique_words"]),
                np.mean(Eco_Abstracts["num_unique_words"]),np.mean(Bio_Abstracts["num_unique_words"])]

Num_Stopwords_Mean = [np.mean(Chem_Abstracts["num_stopwords"]),np.mean(CS_Abstracts["num_stopwords"]),
                np.mean(Eco_Abstracts["num_stopwords"]),np.mean(Bio_Abstracts["num_stopwords"])]

Num_Punctuations_Mean = [np.mean(Chem_Abstracts["num_punctuations"]),np.mean(CS_Abstracts["num_punctuations"]),
                np.mean(Eco_Abstracts["num_punctuations"]),np.mean(Bio_Abstracts["num_punctuations"])]

#Plot Average Number of Words per Abstract
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.figure(figsize=(2,10))
plt.bar(y_pos, Num_Words_Mean)
plt.xticks(y_pos, labels)
plt.legend(fontsize=6)
plt.title('Average Number of Words Per Abstract', fontsize=24)
plt.show()

#Plot Average Number of Unique Words per Abstract
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.figure(figsize=(2,10))
plt.bar(y_pos, Num_Unique_Words_Mean)
plt.xticks(y_pos, labels)
plt.legend(fontsize=6)
plt.title('Average Number of Unique Words Per Abstract', fontsize=24)
plt.show()

#Plot Average Number of Stop Words per Abstract
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.figure(figsize=(2,10))
plt.bar(y_pos, Num_Stopwords_Mean)
plt.xticks(y_pos, labels)
plt.legend(fontsize=6)
plt.title('Average Number of StopWords Per Abstract', fontsize=24)
plt.show()

#Plot Average Number of Punctuations per Abstract
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.figure(figsize=(2,10))
plt.bar(y_pos, Num_Punctuations_Mean)
plt.xticks(y_pos, labels)
plt.legend(fontsize=6)
plt.title('Average Number of Punctuations Per Abstract', fontsize=24)
plt.show()

#Combining all the Training Data
Train = pd.concat([Eco_Abstracts,CS_Abstracts,Bio_Abstracts, Chem_Abstracts], ignore_index=True)

#Combining all the Testing Data
Test =  pd.concat([Eco_Test,CS_Test,Bio_Test, Chem_Test], ignore_index=True)

#Creating X_Train and Y_Train
X_Train = Train.iloc[:,2:6]
Y_Train = Train["Label"]

#Creating X_Test and Y_Test
X_Test = Test.iloc[:,2:6]
Y_Test = Test["Label"]

#Creating First XGBoost Classifier
boost = xgboost.XGBClassifier(n_estimators=800)
boost.fit(X_Train,Y_Train)

#Getting predictions
pred = list(boost.predict(X_Test))
true = list(Y_Test)

#Checking how many errors there are
error = 0
for x in range(len(true)):
    if true[x] != pred[x]:
        error += 1

#Creating confusion matrix
plt.rcParams.update(plt.rcParamsDefault)

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
    

#Finding the 100 most common words across all the abstracts
words = []
for passage in Train["Abstracts"]:
    words.append(re.findall(r'\w+', passage))

words = [j for i in words for j in i]

words=  [w for w in words if (not w.lower() in eng_stopwords and w[0] != '/' and w[0] != 'x' and len(w) != 1)]

cap_words = [word.upper() for word in words]

word_counts = Counter(cap_words)

sorted_counts = OrderedDict(sorted(word_counts.items(), key=itemgetter(1),reverse = True))

top_100 = list(sorted_counts)[0:100]

#Create one hot bag of words encoding to add to the data
max_words = 100 
tokenize = Tokenizer(num_words = max_words, char_level = False)
tokenize.fit_on_texts(words)
x_train = pd.DataFrame(tokenize.texts_to_matrix(Train["Abstracts"]))
x_test = pd.DataFrame(tokenize.texts_to_matrix(Test["Abstracts"]))
X_Train = pd.concat([X_Train,x_train], axis = 1)
X_Test = pd.concat([X_Test,x_test], axis = 1)

#Creating Second XGBoost Classifier
boost = xgboost.XGBClassifier(n_estimators=800)
boost.fit(X_Train,Y_Train)

#Getting predictions
pred = list(boost.predict(X_Test))
true = list(Y_Test)

#Checking how many errors there are
error = 0
for x in range(len(true)):
    if true[x] != pred[x]:
        error += 1

#Creating confusion matrix
plt.rcParams.update(plt.rcParamsDefault)

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
    
