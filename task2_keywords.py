''' module for getting keywords from a title '''
import sys

def keywords(title):
    ''' function that recieves title and return the keywords ''' 
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer

    # turn input to list to be analysed
    title = [title]

    # #Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(title)

    # return an array of keywords from the title
    return tfidf.get_feature_names()

# call function
print(keywords(sys.argv[1]))