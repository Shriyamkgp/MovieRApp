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
vectorizer = vectorizer.fit(clean_train_review)
train_data_features = vectorizer.transform(clean_train_review)
train_data_features #returns a sparse matrix
np.asarray(train_data_features)

#Training the Random Forest classifier
from sklearn.model_selection import GridSearchCV

X = train_data_features
y = label
param_grid = { 
    'n_estimators': [100,200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}
forest_model = RandomForestClassifier(n_estimators = 100)
forest = GridSearchCV(estimator=forest_model, param_grid=param_grid, cv= 5)
forest.fit(train_data_features, label)

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

#Saving the vectorizer and model through pickle
import pickle
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))
pickle.dump(forest, open('forest_clf.pkl','wb'))
