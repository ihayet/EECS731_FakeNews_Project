import numpy as np
import pandas as pd

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from textblob import TextBlob

from functools import reduce

import sys

class Extractor:
    data = ''

    def __init__(self, data):
        self.data = data
        self.data = self.data.dropna()

    def removeSpecialCharacters(self, content):
        return re.sub('[^a-zA-z0-9]+', ' ', str(content))
        
    def tokenize(self, content):
        return word_tokenize(content)

    def lower(self, content):
        return [w.lower() for w in content]

    def removeStopWords(self, content):
        return [w for w in content if w not in stop_words]

    def titleFeatures(self, frame):
        title = str(frame['title'])

        title = self.removeSpecialCharacters(title)
        title = self.tokenize(title)
        title = self.lower(title)
        
        # Sentiment
        textBlob = TextBlob(str(frame['title']))
        frame['title_subjectivity'], frame['title_sentiment_polarity'] = textBlob.sentiment.subjectivity, textBlob.sentiment.polarity
        
        # Words
        frame['n_tokens_title'] = np.log(len(title) + 1)

        return frame

    def textFeatures(self, frame):
        text = str(frame['text'])

        text = self.removeSpecialCharacters(text)
        text = self.tokenize(text)
        text = self.lower(text)
        
        if len(text) == 0:
            # frame['average_token_length'] = 0                
            frame['n_tokens_content'] = 0
            frame['n_unique_tokens'] = 0
            frame['n_non_stop_words'] = 0
            frame['n_non_stop_unique_tokens'] = 0
            frame['global_subjectivity'] = 0.5
            frame['global_sentiment_polarity'] = 0
            frame['global_rate_positive_words'] = 0
            frame['global_rate_negative_words'] = 0
            frame['rate_positive_words'] = 0
            frame['rate_negative_words'] = 0
            frame['avg_positive_polarity'] = 0
            frame['min_positive_polarity'] = 0
            frame['max_positive_polarity'] = 0
            frame['avg_negative_polarity'] = 0
            frame['min_negative_polarity'] = 0
            frame['max_negative_polarity'] = 0          
        else:
            # Sentiment
            textBlob = TextBlob(str(frame['text']))

            frame['global_subjectivity'], frame['global_sentiment_polarity'] = textBlob.sentiment.subjectivity, textBlob.sentiment.polarity

            pos_pol = np.array([TextBlob(str(w)).sentiment.polarity for w in text if TextBlob(str(w)).sentiment.polarity > 0])
            neg_pol = np.array([TextBlob(str(w)).sentiment.polarity for w in text if TextBlob(str(w)).sentiment.polarity < 0])

            if len(pos_pol) == 0:
                frame['global_rate_positive_words'] = 0
                frame['rate_positive_words'] = 0
                frame['avg_positive_polarity'], frame['min_positive_polarity'], frame['max_positive_polarity'] = 0, 0, 0
            else:
                frame['global_rate_positive_words'] = len(pos_pol)/len(text)
                frame['rate_positive_words'] = len(pos_pol)/(len(pos_pol) + len(neg_pol))
                frame['avg_positive_polarity'], frame['min_positive_polarity'], frame['max_positive_polarity'] = np.average(pos_pol), np.min(pos_pol), np.max(pos_pol)

            if len(neg_pol) == 0:
                frame['global_rate_negative_words'] = 0
                frame['rate_negative_words'] = 0
                frame['avg_negative_polarity'], frame['min_negative_polarity'], frame['max_negative_polarity'] = 0, 0, 0
            else:
                frame['global_rate_negative_words'] = len(neg_pol)/len(text)
                frame['rate_negative_words'] = len(neg_pol)/(len(pos_pol) + len(neg_pol))
                frame['avg_negative_polarity'], frame['min_negative_polarity'], frame['max_negative_polarity'] = np.average(neg_pol), np.max(neg_pol), np.min(neg_pol)

            # Words
            total_tokens = len(text)
            frame['n_tokens_content'] = np.log(total_tokens)

            if len(text) == 1:
                frame['average_token_length'] = len(text[0])
            else:
                frame['average_token_length'] = np.log(reduce(lambda x, y: (float(x) + float(y)), list(map(lambda x: len(x), text))))
            
            frame['n_unique_tokens'] = len(set(text))/total_tokens
            text = self.removeStopWords(text)
            frame['n_non_stop_words'] = len(text)/total_tokens
            frame['n_non_stop_unique_tokens'] = len(set(text))/total_tokens
        
        return frame

    def extractionFunction(self, frame):
        frame = self.titleFeatures(frame)
        frame = self.textFeatures(frame)
        
        return frame

    def extractFeatures(self):
        self.data = self.data.apply(self.extractionFunction, axis=1)

        return self.data

    def writeToCSV(self, out_path):
        cols = ['id', 'n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 'n_non_stop_words', 'n_non_stop_unique_tokens', 
                'average_token_length', 'global_subjectivity', 'global_sentiment_polarity', 'global_rate_positive_words',
                'global_rate_negative_words', 'rate_positive_words', 'rate_negative_words', 'avg_positive_polarity',
                'min_positive_polarity', 'max_positive_polarity', 'avg_negative_polarity', 'min_negative_polarity',
                'max_negative_polarity', 'title_subjectivity',
                'title_sentiment_polarity']

        self.data.to_csv(out_path, columns=cols, header=cols, index=False, encoding='utf-8', line_terminator='\n')
    
    def display(self):
        print(data[0:50])

#---------------------------------------------------------------------------------------------------------------------------------------
