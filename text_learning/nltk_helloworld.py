
import nltk
# my nltk_data is somewhere else
nltk.data.path.append('/media/claudio/code-media/Claudio/workspace/python/nltk_data')

from nltk.corpus import stopwords
sw = stopwords.words('english')
print(len(sw))

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("reponsiveness")
stemmer.stem("resp")

import pickle

with open("your_word_data.pkl", "r") as f:
    word_data = pickle.load(f)
    from sklearn.feature_extraction.text import TfidfVectorizer 
    vectorizer = TfidfVectorizer(stop_words = 'english', lowercase=True)
    vectorizer.fit_transform(word_data)
    print(len(vectorizer.get_feature_names()))