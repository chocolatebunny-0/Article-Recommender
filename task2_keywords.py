''' module for getting keywords from a title '''
from sklearn.feature_extraction.text import TfidfVectorizer
def keywords(title):
    ''' function that recieves title and return the keywords '''

    # turn input to list to be analysed
    title = [title]

    # Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # #Construct the required TF-IDF matrix by tokenizing and transforming the data
    tfidf.fit_transform(title)

    # return an string of keywords from the title
    return " ".join(tfidf.get_feature_names())

# call function
print(keywords("Learning Web development at StartNg"))
