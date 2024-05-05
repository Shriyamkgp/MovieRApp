import pandas as pd
import numpy as np
import re 
import nltk
from nltk.corpus import stopwords

class Tokenization():

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        review_text = re.sub("[^a-zA-Z]"," ", review)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return(words)

    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
            raw_sentences = tokenizer.sent_tokenize(review)
            sentences = []
            for raw_sentence in raw_sentences:
                if len(raw_sentence) > 0:
                    sentences.append( Tokenization.review_to_wordlist( raw_sentence, remove_stopwords ))
            return sentences


#model training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

traindata = pd.read_table('labeledTrainData.tsv')
label = traindata['sentiment']
review = traindata['review']

#cleaning our reviews, removing the stopwords
word_to_utility = Tokenization()
clean_train_review = []
for i in range(len(review)):
    clean_train_review.append(' '.join(word_to_utility.review_to_wordlist(review[i])))
clean_train_review

vectorizer = CountVectorizer(analyzer = "word",   tokenizer = None,  preprocessor = None, stop_words = None, max_features = 5000)

#vectorization
train_data_features = vectorizer.fit_transform(clean_train_review)
train_data_features #returns a sparse matrix
np.asarray(train_data_features)

#Training the Random Forest classifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features,label)

#Test Data
testdata = pd.read_csv('test_set_3.csv')
review_test = testdata['review']

# Clean Test Data
word_to_utility = Tokenization()
clean_test_review = []
for i in range(0,len(review)):
    clean_test_review.append(' '.join(word_to_utility.review_to_wordlist(review_test[i],True)))

#transforming into vector
test_data_features = vectorizer.transform(clean_test_review)
np.asarray(test_data_features)
res = forest.predict(test_data_features)
output = pd.DataFrame(data = {"review":review_test, 'sentiment': res})

#drawing confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(testdata['sentiment'],output['sentiment'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()