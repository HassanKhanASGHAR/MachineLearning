# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 00:34:36 2023

@author: dagom
"""
#### analisis de texto 
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import model_selection, svm
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


vectorizer = CountVectorizer()
vectorizer

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus) ## transformar en matriz
X

y=[1, 0, 1, 1 ] ## target variable
#coun_vect = CountVectorizer()

count_array = X.toarray()  ## para ver la matriz en formato no sparse
df = pd.DataFrame(data=count_array,columns = vectorizer.get_feature_names_out())
print(df)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df,y,test_size=0.5)


## df es el equivalente al terms document matrix

analyze = vectorizer.build_analyzer() ## tokenizador tranforma el texto en una lista de palabras
analyze("This is a text document to analyze.") == (
    ['this', 'is', 'text', 'document', 'to', 'analyze'])

#########################


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(corpus)
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print(Tfidf_vect.vocabulary_)
print(Train_X_Tfidf)


SVM = svm.SVC(C=1.0, kernel='linear')
SVM.fit(Train_X,Train_Y)  ## este SVM coge objetos Tfi
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

##########################un dataset mas complicado tuits ##
# Lectura de datos

#tuits = pd.read_csv('C:/Users/dagom/Downloads/svm_phyton/trump_tweets.csv') 
#label=pd.read_csv('C:/Users/dagom/Downloads/svm_phyton/labeled_data.csv') 

# ==============================================================================
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/Estadistica-con-R/master/datos/'
tweets_elon   = pd.read_csv(url + "datos_tweets_@elonmusk.csv")
tweets_edlee  = pd.read_csv(url + "datos_tweets_@mayoredlee.csv")
tweets_bgates = pd.read_csv(url + "datos_tweets_@BillGates.csv")

print('Número de tweets @BillGates: ' + str(tweets_bgates.shape[0]))
print('Número de tweets @mayoredlee: ' + str(tweets_edlee.shape[0]))
print('Número de tweets @elonmusk: ' + str(tweets_elon.shape[0]))

# Se unen los dos dataframes en uno solo
tweets = pd.concat([tweets_elon, tweets_edlee, tweets_bgates], ignore_index=True)

# Se seleccionan y renombran las columnas de interés
tweets = tweets[['screen_name', 'created_at', 'status_id', 'text']]
tweets.columns = ['autor', 'fecha', 'id', 'texto']

# Parseo de fechas
tweets['fecha'] = pd.to_datetime(tweets['fecha'])
tweets.head(3)

# Distribución temporal de los tweets
# ==============================================================================
fig, ax = plt.subplots(figsize=(9,4))

for autor in tweets.autor.unique():
    df_temp = tweets[tweets['autor'] == autor].copy()
    df_temp['fecha'] = pd.to_datetime(df_temp['fecha'].dt.strftime('%Y-%m'))
    df_temp = df_temp.groupby(df_temp['fecha']).size()
    df_temp.plot(label=autor, ax=ax)

ax.set_title('Número de tweets publicados por mes')
ax.legend();

## Paso 1 tokenizar el texto y crear el 
corpus = tweets[['texto']].values.tolist()
i=1
n=len(corpus)
for i in range(n) : 
 corpus[i]=str(corpus[i])

vectorizer = CountVectorizer()
vectorizer

X = vectorizer.fit_transform(corpus) ## transformar en matriz

count_array = X.toarray()
df = pd.DataFrame(data=count_array,columns = vectorizer.get_feature_names_out())
print(df)

y= tweets[['autor']].values.tolist()

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df,y,test_size=0.3)


#Tfidf_vect = TfidfVectorizer(max_features=5000)
#Tfidf_vect.fit(corpus)
#Train_X_Tfidf = Tfidf_vect.transform(Train_X)
#Test_X_Tfidf = Tfidf_vect.transform(Test_X)

#print(Tfidf_vect.vocabulary_)
#print(Train_X_Tfidf)

SVM = svm.SVC(C=1.0, kernel='linear')
SVM.fit(Train_X,np.ravel(Train_Y))
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)





################################################
################################################
## text classification with pipeline !!!!!
import nltk, random
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')


print(len(movie_reviews.fileids()))
print(movie_reviews.categories())
print(movie_reviews.words()[:100])
print(movie_reviews.fileids()[:10])


from sklearn.datasets import load_files
dataset = load_files('C:/Users/dagom/Documents/movie_reviews/', shuffle=False)

dataset.target_names

dataset.filenames

len(dataset.data)

from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC, NuSVC


pipeline = Pipeline([('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
                     ('clf', LinearSVC(C=1000)),
])


from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], }

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
grid_search.fit(docs_train, y_train)

n_candidates = len(grid_search.cv_results_['params'])
for i in range(n_candidates):
    print(i, 'params - %s; mean - %0.2f; std - %0.2f'
             % (grid_search.cv_results_['params'][i],
                grid_search.cv_results_['mean_test_score'][i],
                grid_search.cv_results_['std_test_score'][i]))
    

y_predicted = grid_search.predict(docs_test)

from sklearn import metrics
print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))

cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

########## tex clasification sin pipeline

import nltk, random
from nltk.corpus import movie_reviews
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


print(len(movie_reviews.fileids()))
print(movie_reviews.categories())
print(movie_reviews.words()[:100])
print(movie_reviews.fileids()[:10])



documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.seed(123)
random.shuffle(documents)


print('Number of Reviews/Documents: {}'.format(len(documents)))
print('Corpus Size (words): {}'.format(np.sum([len(d) for (d,l) in documents])))
print('Sample Text of Doc 1:')
print('-'*30)
print(' '.join(documents[0][0][:50])) # first 50 words of the first document


## Check Sentiment Distribution of the Current Dataset
from collections import Counter
sentiment_distr = Counter([label for (words, label) in documents])
print(sentiment_distr)


from sklearn.model_selection import train_test_split
train, test = train_test_split(documents, test_size = 0.33, random_state=42)
## Sentiment Distrubtion for Train and Test
print(Counter([label for (words, label) in train]))
print(Counter([label for (words, label) in test]))

X_train = [' '.join(words) for (words, label) in train]
X_test = [' '.join(words) for (words, label) in test]
y_train = [label for (words, label) in train]
y_test = [label for (words, label) in test]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tfidf_vec = TfidfVectorizer(min_df = 10, token_pattern = r'[a-zA-Z]+')
X_train_bow = tfidf_vec.fit_transform(X_train) # fit train
X_test_bow = tfidf_vec.transform(X_test) # transform test

print(X_train_bow.shape)
print(X_test_bow.shape)

from sklearn import svm

model_svm = svm.SVC(C=8.0, kernel='linear')
model_svm.fit(X_train_bow, y_train)

from sklearn.model_selection import cross_val_score
model_svm_acc = cross_val_score(estimator=model_svm, X=X_train_bow, y=y_train, cv=5, n_jobs=-1)
model_svm_acc

model_svm.predict(X_test_bow[:10])
#print(model_svm.score(test_text_bow, test_label))