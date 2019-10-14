#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mysql-connector-python ')


# In[2]:


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


# In[3]:


#his code is on this cell
def keywords(title):
    ''' function that recieves title and return the keywords ''' 
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    title = [title]
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit_transform(title)
        
    return " ".join(tfidf.get_feature_names())

# call function
keywords("Learning Web development at StartNg")


# In[49]:


mydb = mysql.connector.connect(host="remotemysql.com",
                              user="8SawWhnha4",
                              passwd="zFvOBIqbIz",
                              database="8SawWhnha4")
engine = create_engine('mysql+mysqlconnector://8SawWhnha4:zFvOBIqbIz@remotemysql.com/8SawWhnha4')

df = pd.read_sql_query('select title from posts where id>20 LIMIT 10', engine)

df['title'] = df['title'].replace({"'ll": " "}, regex=True)

df['keywords'] = None

for i in range(len(df)):
    df.iat[i, df.columns.get_loc("keywords")] = keywords(df.iat[i,0])
    
art_title= [df.iat[i,1] for i  in range(len(df))]
tfidf = TfidfVectorizer(stop_words='english')
t_matrix = tfidf.fit_transform(art_title)

# Create a KMeans object with 5 clusters and save as K_M
K_M = KMeans(n_clusters=5)
#unpack the dataframe and matrix

# Fit the k-means object with tfidf_matrix
K_M.fit(t_matrix)

CLUSTERS = K_M.labels_.tolist()

# Create a column cluster to denote the generated cluster for each article
df["CLUSTER"] = CLUSTERS

# Display number of articles  per cluster (clusters from 0 to 4)
df['CLUSTER'].value_counts() 


# In[50]:


#measuring distance
SIM_DIS = 1 - cosine_similarity(t_matrix)


# In[54]:


df


# In[8]:


#to save this model 
#FILE_NAME ='built_model.sav'
#pickle.dump(K_M, open(FILE_NAME, 'wb'))


# In[9]:


#to load saved model later
#GET_SAVED = pickle.load(open(FILE_NAME, 'rb'))
#DISPLAY = GET_SAVED.score()
#print(DISPLAY)


# In[55]:



COLS = ['title' ,'keywords']
df['in_string'] = df.loc[:,COLS].apply(lambda x: x.dropna().tolist(), 1)
df['rank'] = df.reset_index().index
df


# In[47]:


df.in_string.head()


# In[56]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.cluster.hierarchy import linkage, dendrogram

MERGINGS = linkage(SIM_DIS, method='complete')

TITLE_DENGROGRAM = dendrogram(MERGINGS,
               labels=[x for x in df["title"]],
               leaf_rotation=90,
               leaf_font_size=16,
)

fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(60, 15)

# Show the plotted dendrogram
plt.show()


# In[10]:


#making a dictionary that held the most similar titles
similar_articles = {}

for title in MERGINGS:
    title_0 = int(title[0])
    title_1 = int(title[1])
    similar_articles[title_0] = title_1
    
rated_title = []
similar_for_rated = []

for a in similar_articles:
    rated_title.append(a)
    similar_for_rated.append(similar_articles[a])


# In[95]:


#predict method
def show_most_similar_article():
    article_title = input('Please Enter a movie title: ').strip()
    df['title_lower'] = df['title'].apply(lambda x: x.lower())
    
    #articles['title_lower'] = str(articles['title']).lower()  
    #return 'article most similar to {} is {}'.format(article_title, similar_movies[_ClUSTER])
    if any(df['title_lower'] == article_title.lower()):
        title_df = df[df['title_lower'] == article_title]
    else:
        return "Article does not exist. Please check your spelling and Capitalisations"
    
    rank = title_df['rank'].astype(int)
    
    if 'rank' in rated_title:
        sim_title_df = df[df['rank'] == similar_articles[rank]]
        
        global sim_article
        sim_article = sim_title_df.title.values
      
    elif 'rank' in similar_for_rated:
        indx = similar_for_rated.index(rank)
        sim_title_df = df[df['rank'] == rated_title[indx]]
        sim_article = sim_title_df.title.values
    if sim_article.size > 0:
        sem = sim_article[0]
    else:
        sem = 'sorry no articles Available'
        
    return 'Most Similar article to {} is {}'.format(article_title, sem)
        


# In[97]:


show_most_similar_article()

