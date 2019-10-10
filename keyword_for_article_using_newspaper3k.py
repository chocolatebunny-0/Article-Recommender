from newspaper import Article 
import nltk
nltk.download('punkt')

#A new article from TOI 
url = "https://thenationonlineng.net/obaseki-swears-in-iyase-as-commissioner-for-special-duties/"

#For different language newspaper refer above table 
article = Article(url, language="en") # en for English 

#To download the article 
article.download() 

#To parse the article 
article.parse() 

#To perform natural language processing ie..nlp 
article.nlp() 

#To extract title 
print("Article's Title:") 
print(article.title) 
print("n") 

#To extract text 
print("Article's Text:") 
print(article.text) 
print("n") 

#To extract summary 
print("Article's Summary:") 
print(article.summary) 
print("n") 


#To extract keywords 

def keyword():
  print ("keyword in article")
  print (article.keywords)
  
keyword()