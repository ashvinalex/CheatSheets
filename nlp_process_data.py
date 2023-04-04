import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class DataWranglerAndPreparer(BaseEstimator, TransformerMixin):
    
    def _init_(self, stop_word_lang="english"):
        self.stop_word_lang = stop_word_lang
        self.data_transformed = None
        
    def fit(self, X, y=None):
        self.data_transformed = X
        return self
    
    def transform(self, X, y=None):
        X = self.only_letters(X)
        X = self.tokenization(X)
        X = self.remove_stop_words(X)
        X = self.pos(X)
        X = self.lemmatization(X)
        self.data_transformed = X
        return X
    
    def only_letters(self, X: pd.Series) -> pd.Series:
        return X.apply(lambda text: re.sub(r"[^a-z\s]", "", text.lower()))

    def tokenization(self, X: pd.Series) -> pd.Series:
        return X.apply(lambda text: word_tokenize(text))
    
    def stop_words(self, document: list, language: str) -> pd.DataFrame:
        stop_words_list = stopwords.words(language)
        token_list = []
        for token in document:
            if token not in stop_words_list:
                token_list.append(token)
        return token_list
    
    def remove_stop_words(self, X: pd.Series) -> pd.Series:
        return X.apply(lambda text: self.stop_words(text, self.stop_word_lang))
    
    def pos(self, X: pd.Series) -> pd.Series:
        return X.apply(lambda text: nltk.pos_tag(text))
    
    def get_pos(self, tag):

        if tag.startswith("J"):
            return "a"
        elif tag.startswith("V"):
            return "v"
        elif tag.startswith("N"):
            return "n"
        elif tag.startswith("R"):
            return "r"
        else:
            return "n"
        
    def lematize(self, lem: WordNetLemmatizer, document: list) -> list:
        lemmatized_document = []
        for token in document:
            pos = self.get_pos(token[1])
            lemmatized_document.append(lem.lemmatize(token[0], pos=pos))
        return " ".join(lemmatized_document)

    def lemmatization(self, X: pd.Series) -> pd.Series:
        lem = WordNetLemmatizer()
        return X.apply(lambda text: self.lematize(lem=lem, document=text)
