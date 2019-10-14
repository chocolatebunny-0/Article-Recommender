#!/usr/bin/env python
# coding: utf-8

# In[45]:


#get_ipython().system('pip install mysql-connector-python ')


# In[46]:


#get_ipython().system('pip install flask')


# In[5]:


import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import nltk
from nltk.stem.snowball import SnowballStemmer
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import pickle
nltk.download('punkt')


# In[7]:


mydb = mysql.connector.connect(host="remotemysql.com",
                              user="8SawWhnha4",
                              passwd="zFvOBIqbIz",
                              database="8SawWhnha4")

engine = create_engine('mysql+mysqlconnector://8SawWhnha4:zFvOBIqbIz@remotemysql.com/8SawWhnha4')


# In[8]:


POSTS = pd.read_sql_query('select * from posts', engine)
POSTS.head()


# In[10]:


POSTS['title'] = POSTS['title'].astype(str) +"\n"


# In[11]:


SENT_TOKENIZED =  [sent for sent in nltk.sent_tokenize("""
                         What i have learnt so far on HTML.""")]


# In[12]:


WORD_TOKENIZED = [word for word in nltk.word_tokenize(SENT_TOKENIZED[0])]
FILTERED = [word for word in WORD_TOKENIZED if re.search('[a-zA-Z]', word)]


# In[13]:


#FILTERED 
STEMMER = SnowballStemmer("english")
STEMMED_WORDS = [STEMMER.stem(word) for word in FILTERED]
print("After stemming:   ", STEMMED_WORDS)


# In[14]:


def tokenize_plus_stem(text):
    tokens = [y for x in nltk.sent_tokenize(text) for y in nltk.word_tokenize(x)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    stems = [STEMMER.stem(word) for word in filtered_tokens]
    return stems
WORDS_STEMMED = tokenize_plus_stem
(" What i have learnt so far on HTML.")
print(WORDS_STEMMED)


# In[15]:


TFIDF_MYOBJECT = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_plus_stem,
                                 ngram_range=(1, 3))
TFIDF_MATRIX = TFIDF_MYOBJECT.fit_transform([x for x in POSTS["title"]])
print(TFIDF_MATRIX.shape)


# In[35]:


#his code is on this cell
def keywords(title):
    ''' function that recieves title and return the keywords ''' 
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    # turn input to list to be analysed
    title = [title]

    # #Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(title)

    # return an array of keywords from the title
    return tfidf.get_feature_names()


# In[36]:



# Create a KMeans object with 5 clusters and save as K_M
K_M = KMeans(n_clusters=5)

# Fit the k-means object with tfidf_matrix
K_M.fit(TFIDF_MATRIX)

CLUSTERS = K_M.labels_.tolist()

# Create a column cluster to denote the generated cluster for each article
POSTS["CLUSTER"] = CLUSTERS

# Display number of articles  per cluster (clusters from 0 to 4)
POSTS['CLUSTER'].value_counts() 


# In[37]:


#measuring distance
DIS_SIM = 1 - cosine_similarity(TFIDF_MATRIX)


# In[19]:


MERGINGS = linkage(DIS_SIM, method='complete')
POSTS_DENDROGRAM = dendrogram(MERGINGS,
                               labels=[x for x in POSTS["title"]],
                               leaf_rotation=90,
                               leaf_font_size=16,)

FIG = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
FIG.set_size_inches(100, 20)
plt.show()


# In[171]:


#to save this model 
FILE_NAME ='built_model.sav'
pickle.dump(K_M, open(FILE_NAME, 'wb'))


# In[174]:


#to load saved model later
#GET_SAVED = pickle.load(open(FILE_NAME, 'rb'))
#DISPLAY = GET_SAVED.score()
#print(DISPLAY)


# In[175]:


print(POSTS.dtypes)


# In[176]:


'''A function to predict the cluster of any 
description put into it'''

def cluster_pred(str_inp):
    fresh = tfidf.transform(list(str_inp))
    prediction = K_M.predict(fresh)
    return prediction
COLS = ['user_id' ,'title' ,'tags']
POSTS['in_string'] = POSTS.loc[:,COLS].apply(lambda x: x.dropna().tolist(), 1)

#Assign categories to each title based on descripion vector in a new DF
#create a new column called "ClusterPred"
POSTS.head()

 


# In[182]:


COLS = ['user_id' ,'title' ,'tags']
def recomm_title(str_inp):
    title_df = POSTS.loc[POSTS['user_id'] == str_inp]
    title_df['in_string'] = title_df.loc[:,COLS].apply(lambda x: x.dropna().tolist(), 1)
    str_inp = list(title_df['in_string'])
   # Based on the above prediction 10 random titles will be recommended from the whole data-frame
    title_df = POSTS.loc[POSTS['CLUSTER'] == predict_inp]
    title_df = title_df.in_string.sample(10)
    return list(title_df)
POSTS.in_string.sample(10)

