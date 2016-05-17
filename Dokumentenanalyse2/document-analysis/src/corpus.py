
import nltk

class CorpusLoader(object):

    @staticmethod
    def load():
        # ##################
        # Check for corpus or download
        # ##################
        try:
            nltk.corpus.brown.words()
            nltk.corpus.stopwords.words()
        except LookupError:
            print 'Download brown and stopword corpus'
            nltk.download()
        return
    
    @staticmethod
    def brown_corpus():
        return nltk.corpus.brown
    
    @staticmethod
    def stopwords_corpus():
        stopwords_en = nltk.corpus.stopwords.words('english')
        return stopwords_en

