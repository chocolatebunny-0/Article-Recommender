""" Function that takes in title of an article and predicts similar articles to users """
# **Step 1 - Train the engine.**
#
# Create a TF-IDF matrix of unigrams, bigrams, and trigrams for each product.
# The 'stop_words' param tells the TF-IDF module to ignore common english words like 'the', etc.
#
# Then we compute similarity between all articles using SciKit Leanr's linear_kernel
# (which in this case is equivalent to cosine similarity).
#
# Iterate through each article's similar articles and store the 100 most-similar.
# You could show more than 100, your choice.
#
# Similarities and their scores are stored in a dictionary as a list of Tuples,
# indexed to their post id

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import mysql.connector
from sqlalchemy import create_engine

#loading the dataset
MYDB = mysql.connector.connect(host="remotemysql.com", user="8SawWhnha4", passwd="zFvOBIqbIz",
                               database="8SawWhnha4")

ENGINE = create_engine('mysql+mysqlconnector://8SawWhnha4:zFvOBIqbIz@remotemysql.com/8SawWhnha4')

#fetching the tables in the dataset
DB_CURSOR = MYDB.cursor()
DB_CURSOR.execute('show tables')
for table in DB_CURSOR:
    print(table)

#checking out the post table
POSTS = pd.read_sql_query('select * from posts', ENGINE)
POSTS.drop(['user_id', 'tags', 'slug', 'created_at', 'updated_at', 'image',
            'status_id', 'action', 'post_id'], axis=1, inplace=True)
POSTS.rename(columns={"id":"post_id"}, inplace=True)
POSTS.head(50)

TF = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
TFIDF_MATRIX = TF.fit_transform(POSTS['title'])

COSINE_SIMILARITIES = linear_kernel(TFIDF_MATRIX, TFIDF_MATRIX)

RESULTS = {}

for idx, row in POSTS.iterrows():
    similar_indices = COSINE_SIMILARITIES[idx].argsort()[:-100:-1]
    similar_items = [(COSINE_SIMILARITIES[idx][i], POSTS['post_id'][i]) for i in similar_indices]

    # First post is the post itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, post_id)
    RESULTS[row['post_id']] = similar_items[1:]
print('done!')

def post(p_id):
    """ Function to get an article title from the title field,
    given a post ID 
    Args: The post id
    Arg type: Integer
    Returns: A particular post title with id = post_id"""
    return POSTS[POSTS['post_id'] == p_id]['title'].tolist()[0].split(' - ')[0]


def recommend(post_id, num):
    """ Function to reads the results out of the dictionary.
    Args: post id and number of recommendations required
    Arg type: Integer, Integer
    Returns: Dictionary of similar titles and score"""
    print("Recommending " + str(num) + " articles similar to " + post(post_id) + "...")
    print("--------------------------------------------------------------------------")
    recs = RESULTS[post_id][:num]
    for rec in recs:
        print("Recommended: " + post(rec[1]) + " (score:" + str(rec[0]) + ")")

# Just plug in any post id here (we have about 800 posts in the dataset), and the number of recommendations you want (1-99)
# You can get a list of valid post IDs by evaluating the variable 'POSTS'

recommend(post_id=33, num=5)
