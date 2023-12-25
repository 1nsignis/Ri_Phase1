from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

#Vector Space Model
class VSMindex():

    def __init__(self, corpus, _ngram_range=(1,1), _analyzer='word', _stop_words = None):
        self.index  = TfidfVectorizer(ngram_range=_ngram_range, analyzer=_analyzer, stop_words = _stop_words)
        self.doc_matrix = self.index.fit_transform(corpus)


    def search(self, query : str):
        query_vec = self.index.transform([query])
        doc_scores = 1 - pairwise_distances(self.doc_matrix, query_vec, metric='cosine')
        return doc_scores.flatten()

#Language Model
class LMJMindex():
    
    def __init__(self, corpus, _ngram_range=(1,1), _analyzer='word', _stop_words = None):
        self.vectorizer  = CountVectorizer(ngram_range=_ngram_range, analyzer=_analyzer, stop_words = _stop_words)
        doc_indexes = self.vectorizer.fit_transform(corpus).todense()
        col_index   = np.sum(doc_indexes, axis=0)
        self.prob_term_col = col_index / np.sum(col_index)
        self.prob_term_doc = doc_indexes / np.sum(doc_indexes, axis=1)


    def set_params(self, params):
        if 'lambda' in params:
            self.lbd = params['lambda']
            self._log_lmjm = np.log(self.lbd * self.prob_term_doc + (1-self.lbd) * self.prob_term_col)
            print("LMJM lambda ", self.lbd)

    def search(self, query : str):
        query_vec = self.vectorizer.transform([query])
        doc_scores = query_vec.dot(self._log_lmjm.T)

        return np.asarray(doc_scores).flatten()


