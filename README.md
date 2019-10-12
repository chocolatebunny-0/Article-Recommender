# Article Recommender 
   This repository contains all the functions, model and url required to complete the Article Recommender task given.

### DataSet used  for the task:
Lucid blog Dataset 

## Task 1 - Article Keywords 
  This code creates a function that returns the keywords from any article.
### Depenencies required
Newspaper3k library
BeautifulSoup
###  Testing
  A function was created “keyword”, once it’s called.  It’s displays the keyword in the url you sent.
  
## Task 2 - Article Title Keywords
  This code creates  function that takes in the title of an article, tokenize and remove stopwords from the title and returns the keywords from the title. It is created as a cli script that takes a title(string) as a parameter and return a list of the titles keywords
### Dependencies required - 
sklearn module  
### Testing
To test it type python task2_keywords.py "article title"

  
## Task 3 - Article Keywords model
  This model understands the cluster from task 2 
  k-means clustering is used to get clusters from the defined function.
### Testing 
When you run the cell it should show  the number of clusters each article has based on the already selected number of articles which is limited to 10.
To save the model, the user can comment out  and run the cell, the model will be saved on the user's disk... the user can also comment out the cell for loading the saved model, run and get the saved model.
   
## Task 4 - Article Title Recommender Function
  This project creates a function thats takes in the title of an article and predicts similar articles to the reader based on the title of the article. tf-idf is used to identify keywords and remove stopwords from post titles, applied cosine similarities to calculate the frequency of likelihood of such keywords appearing in other titles using their post_id as pointers
Thereafter 2 functions were created, one to fetch post_ids which will be used in the second function to recommend similar articles
Function post- has one parameter that's post_id
Function recommend - has 2 parameters post_id and num i.e. number of recommendations

 
## Task 5 - Article Recommender
  A generator that yields similar articles to users based in the same cluster.
  the code provided is meant to define a generatorthat yields similar articles to users based in the same cluster.
Very much aware of the use of K-means for clustering,there is need to recommend articles to those in various 
clusters.So we first create generator which predicts the cluster of any title inputted into it. We then create a recommendation generator to predict the title category of the inputted query of CLUSTER column in the table. We recommend few 
random titles from (a new column generated for storing predicted categories from our trained model)
  
## Task 6 - Article Recommender ( using CLI scrript)
  A CLI script that yields similar articles to users based in the same cluster. 
  
## Task 21 -Deploying article recommender on Heroku
   url- 
   
## Acknowledgements
   * Hat tip to everyone who contributed to the success of this task 
  
