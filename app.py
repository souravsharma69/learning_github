# importing libraries
import re
import string
import nltk                                
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTKpip
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.stem import WordNetLemmatizer    # module for Lemmatization
import pickle
from nltk.tokenize import TweetTokenizer
import sklearn
nltk.download('wordnet')
nltk.download('omw-1.4')
import warnings
warnings.filterwarnings('ignore')

#function 1
def process_tweet(tweet):
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            lemma_word = lemmatizer.lemmatize(word,pos='v')  # stemming word
            tweets_clean.append(lemma_word)

    return tweets_clean


#function 2
def list_to_string(lst):
    return ' '.join(lst)



#loading vectorizer 
with open(r'C:\Users\sony\Downloads\vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)


# Load the saved model---->desrialaiztion
with open(r'C:\Users\sony\Downloads\logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)



#new_tweet = "I hate covid because it is killing so many people #KillCorona"
new_tweet = " i love the work of this government"

#1st function call
tweet_list = process_tweet(new_tweet)

#2nd function call
final_tweet = list_to_string(tweet_list)

#vectorization
final_vector = loaded_vectorizer.transform([final_tweet])

#prediction
final_result = loaded_model.predict(final_vector)
print(list(final_result)[0])



#new code 
print("This is Testing")