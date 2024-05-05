import pandas as pd
import numpy as np
import re 
import nltk
from nltk.corpus import stopwords
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Movie Review Sentiment Test \n
""")



uploaded_file = st.file_uploader("Choose a file",type=['txt'])

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    review = stringio.read()
else:
    st.write("""
            OR
            """)
    review = st.text_input("Enter Review:")


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

def action(stri):
    load_forest = pickle.load(open('forest_clf.pkl', 'rb'))
    load_vect = pickle.load(open('vectorizer.pkl', 'rb'))
    word_to_utility = Tokenization()
    cleaned = []
    cleaned.append(' '.join(word_to_utility.review_to_wordlist(stri,True)))
    features = load_vect.transform(cleaned)
    np.asarray(features)
    res = load_forest.predict(features)
    return res


if st.button('  CHECK   '):
    res = action(review)

    st.write("""
             \n
             \n
         ### Sentiment:
         """)

    if res[0] == 1:
        st.write('Positive')
    elif res[0] == 0:
        st.write('Negative')
    else:
        st.write('Error')
    
    st.write("""
         ### Word Cloud
         """)
    wc = WordCloud()
    wc.generate(review)
    plt.imshow(wc)
    plt.axis('off')
    st.pyplot()



