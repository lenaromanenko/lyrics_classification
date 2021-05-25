"""
Pipeline for building a CLI
"""
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Packages for tokenization and stop-words:
#nltk.download('punkt')
#nltk.download('stopwords')

ARTIST1 = 'Frank Sinatra'
ARTIST2 = 'Chris Rea'
LYRICS = []

def train_model(LYRICS):
    num = 50
    def collect_song_links(artist, site = 'metrolyrics'):
        """
        Given some artist and site name, 
        collects links for all the artist's songs 
        and returns them
        """
        links = []  
        if site == 'metrolyrics':
            artist = artist.replace(' ', '-')
            url = "https://www.metrolyrics.com/" + artist + "-lyrics.html"
            response = requests.get(url)
            if response.status_code != 200:
                print('Sorry try again.')
            soup = BeautifulSoup(markup = response.text, features = "lxml")
            for td in soup.find_all('td'):
                if td.a is not None:
                    links.append(td.a.get('href'))  
        elif site == 'lyrics':
            print('Feature not available.')
        return links, artist

    def get_songs_lyrics(links, artist_name, num):
        """
        Given a list of song urls,
        returns a list of the lyrics
        """
        lyrics = []
        for li in links[:num]:
            response = requests.get(li)
            soup = BeautifulSoup(markup = response.text, features = "lxml")
            lyrics_section = soup.find(attrs = {'id':'lyrics-body-text'})
            lyrics_chunk = []
            for verse in lyrics_section.find_all('p', class_ = 'verse'):
                lyrics_chunk.append(verse.text)

            lyrics.append((' '.join(lyrics_chunk))) #artist_name
        return lyrics

    def main(artist, num):
        """ 
        Given artist and num, combines two functions together 
        """
        links, artist = collect_song_links(artist)
        results = get_songs_lyrics(links, artist, num)
        return results
    
    LABELS = [ARTIST1] * num + [ARTIST2] * num
    print("Getting the text...") 
    CORPUS = main(ARTIST1, num) + main(ARTIST2, num)
    print("...done!")
    CORPUS = pd.Series(CORPUS)

    def clean_data(column:pd.Series) -> pd.Series:
        """
        Given df[column], removes punctuation, 
        nums, mult-dots, sq-brackets, 
        replaces new lines with white spaces etc.
        """
        column = column.copy()
        column = column.str.replace(r"[(),:!?@&\'\`\"\_]", "")
        column = column.str.replace(r"[\n]", " ")
        column = column.str.replace(r"[...]", "")
        column = column.str.replace(r"[\d]", "") 
        column = column.str.replace(r"[-]", "") 
        column = column.str.replace(r"[", "")
        column = column.str.replace(r"]", "")
        column = column.str.lower()
        return column

    CORPUS = clean_data(CORPUS)
    
    # 1. Tokenizing text with spaCy
    def spacy_cleaner(document):
        """ 
        Tokenizing the text with spaCy and saving cleaned text
        """
        nlp = spacy.load('en_core_web_md') # the medium-sized model with word vecotrs
        tokenized_doc = nlp(document)
        processed_text_list = []

        for token in tokenized_doc:
            if not token.is_stop and token.is_alpha:
                processed_text_list.append(token.lemma_)            
        return processed_text_list
        
        lyrics_clean = []
        for i in range(len(CORPUS)):
            clean_corpus = spacy_cleaner(CORPUS[i])
            lyrics_clean.append(clean_corpus)
    
        lyrics_final = [' '.join(x) for x in lyrics_clean]
    
    # 2. Cleaning the text with NLTK
    def clean_text_NLTK(text):
        """
        Given text, converts it to lower case, splits up the words,
        and removes all the punctuations and stop words
        """
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(str(text).lower())
        filtered_words = []
        for word in words:
            if word not in stop_words and word.isalpha(): 
                filtered_words.append(word)
        return " ".join(filtered_words)

        cleaning_corpus = []
        for song in CORPUS:
            cleaned_lyrics = clean_text(song)
            cleaning_corpus.append(cleaned_lyrics)

    def train_model_NB(X_train, y_train):
        """
        Putting TfidfVectorizer (TF-IDF) and 
        Naive Bayes algorithm for 
        multinomially distributed data (MultinomialNB) in a pipeline
        """
        pipeline = make_pipeline(TfidfVectorizer(max_features = 1000, min_df = 2, max_df = 0.5, ngram_range = (1,2), stop_words = 'english'),
                                MultinomialNB(alpha = 0.1))
        pipeline.fit(X_train, y_train)
        return pipeline

    X_train, X_test, y_train, y_test = train_test_split(CORPUS, LABELS, test_size = 0.2, random_state = 42)
    pipeline = train_model_NB(X_train, y_train)

    prediction = pipeline.predict([LYRICS])
    probability = pipeline.predict_proba([LYRICS])

    print()
    print("The interpret is:")
    print(prediction)
    print()
    print("The probability is:")
    print(probability.max().round(2))