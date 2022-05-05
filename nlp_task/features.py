from collections import OrderedDict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        words = set()
        for text in X:
            words = words.union(text.split())
            
            
        word_count = {word: 0 for word in words}
        for text in X:
            for word in text.split():
                word_count[word] += 1

        self.bow = [
            word
            for word, count in sorted(
                list(word_count.items()),
                reverse=True,
                key=lambda pair: pair[1],
            )[: self.k]
        ]

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = np.zeros(self.k)
        for word in text.split():
            if word in self.bow:
                result[self.bow.index(word)] += 1

        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        words =set()
        for text in X:
            words = words.union(text.split())

        word_count = {word: 0 for word in words}
        for text in X:
            for word in text.split():
                word_count[word] += 1
                
        self.top_k_words = (
            [
                word
                for word, count in sorted(
                    list(word_count.items()),
                    reverse=True,
                    key=lambda pair: pair[1],
                )[: self.k]
            ]
            if self.k is not None
            else words
        )

        df = {word: 0 for word in self.top_k_words}
        for word in words:
            for text in X:
                if word in text.split():
                    df[word] += 1
        for word in words:
            self.idf[word] = np.log(len(X) / df[word])

        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        result = np.zeros(self.k)
        for word in list(set(text.split()) & set(self.top_k_words)):
            result[list(self.idf.keys()).index(word)] = (
                text.split().count(word) * self.idf[word]
            )

        return np.array(result, "float32")


    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        res = np.stack([self._text_to_tf_idf(text) for text in X])

        if self.normalize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            res = scaler.fit_transform(res)

        return res
