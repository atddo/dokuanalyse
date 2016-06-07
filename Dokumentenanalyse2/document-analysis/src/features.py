import string
import numpy as np
from collections import defaultdict
from corpus import CorpusLoader
from nltk.stem.porter import PorterStemmer # IGNORE:import-error

from operator import itemgetter

class AbsoluteTermFrequencies(object):
    """Klasse, die zur Durchfuehrung absoluter Gewichtung von Bag-of-Words
    Matrizen (Arrays) verwendet werden kann. Da Bag-of-Words zunaechst immer in
    absoluten Frequenzen gegeben sein muessen, ist die Implementierung dieser
    Klasse trivial. Sie wird fuer die softwaretechnisch eleganten Unterstuetzung
    verschiedner Gewichtungsschemata benoetigt (-> Duck-Typing).
    """
    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.

        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).

        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen
                je Dokument (zeilenweise).
        """
        # Gibt das NumPy Array unveraendert zurueck, da die Bag-of-Words Frequenzen
        # bereits absolut sind.
        return bow_mat

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'absolute'


class RelativeTermFrequencies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen
    Bag-of-Words Matrizen (Arrays) in relative Frequenzen.
    """
    @staticmethod
    def weighting(bow_mat):
        """Fuehrt die relative Gewichtung einer Bag-of-Words Matrix (relativ im
        Bezug auf Dokumente) durch.

        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).

        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen
                je Dokument (zeilenweise).
        """
        abs_words_mat = np.sum(bow_mat, axis = 1, dtype = float).reshape(bow_mat.shape[0], 1)
        bow_mat = np.divide(bow_mat, abs_words_mat)

        return bow_mat

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'relative'

class RelativeInverseDocumentWordFrequecies(object):
    """Realisiert eine Transformation von in absoluten Frequenzen gegebenen
    Bag-of-Words Matrizen (Arrays) in relative - inverse Dokument Frequenzen.
    """
    def __init__(self, vocabulary, category_wordlists_dict):
        """Initialisiert die Gewichtungsberechnung, indem die inversen Dokument
        Frequenzen aus dem Dokument Korpous bestimmt werden.

        Params:
            vocabulary: Python Liste von Woertern (das Vokabular fuer die
                Bag-of-Words).
            category_wordlists_dict: Python dictionary, das zu jeder Klasse (category)
                eine Liste von Listen mit Woertern je Dokument enthaelt.
                Siehe Beschreibung des Parameters cat_word_dict in der Methode
                BagOfWords.category_bow_dict.
        """
        cat_bow_dic = BagOfWords(vocabulary).category_bow_dict(category_wordlists_dict)

        all_documents = 0
        for key in category_wordlists_dict:
            all_documents = all_documents + len(category_wordlists_dict[key])

        terms_per_document = []
        terms_per_category = []
        cat_bow_dic_copy = cat_bow_dic
        for key in cat_bow_dic_copy:
            for line in cat_bow_dic_copy:
                for term in line:
                    if term > 0:
                        term = 1
            terms_per_category = np.sum(cat_bow_dic_copy[key], axis=0)
            terms_per_document.append(terms_per_category)
        terms_per_document = np.sum(terms_per_document, axis=0)

        print terms_per_category

        relative_inverse_dic = {}
        for key in cat_bow_dic:
            inverse_dic_mat = []
            for line in cat_bow_dic[key]:
                term_per_all_documents = all_documents / terms_per_document
                
                print line / line.shape[0]
                relative_inverse_line = line / line.shape[0] * np.log(all_documents / terms_per_document)
                inverse_dic_mat.append(relative_inverse_line)
            relative_inverse_dic[key] = inverse_dic_mat

        return relative_inverse_dic

    def weighting(self, bow_mat):
        """Fuehrt die Gewichtung einer Bag-of-Words Matrix durch.

        Params:
            bow_mat: Numpy ndarray (d x t) mit Bag-of-Words Frequenzen je Dokument
                (zeilenweise).

        Returns:
            bow_mat: Numpy ndarray (d x t) mit *gewichteten* Bag-of-Words Frequenzen
                je Dokument (zeilenweise).
        """
        raise NotImplementedError('Implement me')

    def __repr__(self):
        """Ueberschreibt interne Funktion der Klasse object. Die Funktion wird
        zur Generierung einer String-Repraesentations des Objekts verwendet.
        Sie wird durch den Python Interpreter bei einem Typecast des Objekts zum
        Typ str ausgefuehrt. Siehe auch str-Funktion.
        """
        return 'tf-idf'


class BagOfWords(object):
    """Berechnung von Bag-of-Words Repraesentationen aus Wortlisten bei
    gegebenem Vokabular.
    """

    def __init__(self, vocabulary, term_weighting=AbsoluteTermFrequencies()):
        """Initialisiert die Bag-of-Words Berechnung

        Params:
            vocabulary: Python Liste von Woertern (das Bag-of-Word Vokabular)
            term_weighting: Objekt, das die weighting(bow_mat) Methode implemeniert.
                Optional, verwendet absolute Gewichtung als Default.
        """
        self.__vocabulary = vocabulary
        self.__term_weighting = term_weighting
        self.__lookupindex = {}
        for (idx, term) in enumerate(self.__vocabulary):
            self.__lookupindex[term] = idx

    def category_bow_dict(self, cat_word_dict):
        """Erzeugt ein dictionary, welches fuer jede Klasse (category)
        ein NumPy Array mit Bag-of-Words Repraesentationen enthaelt.

        Params:
            cat_word_dict: Dictionary, welches fuer jede Klasse (category)
                eine Liste (Dokumente) von Listen (Woerter) enthaelt.
                cat : [ [word1, word2, ...],  <--  doc1
                        [word1, word2, ...],  <--  doc2
                        ...                         ...
                        ]
        Returns:
            category_bow_mat: Ein dictionary mit Bag-of-Words Matrizen fuer jede
                Kategory. Eine Matrix enthaelt in jeder Zeile die Bag-of-Words
                Repraesentation eines Dokuments der Kategorie. (d x t) bei d
                Dokumenten und einer Vokabulargroesse t (Anzahl Terme).
        """
        category_bow_mat = {}
        gesamtliste = []
        for (key, value) in cat_word_dict.iteritems():
            keylist = []
            arr = np.zeros((len(value), len(self.__vocabulary)))

            for idx, word_list in enumerate(value):
                for k in word_list:
                    if k in self.__lookupindex:
                        arr[idx, self.__lookupindex[k]] +=1

            category_bow_mat[key] = self.__term_weighting.weighting(arr)

        return category_bow_mat


    @staticmethod
    def most_freq_words(word_list, n_words=None):
        """Bestimmt die (n-)haeufigsten Woerter in einer Liste von Woertern.

        Params:
            word_list: Liste von Woertern
            n_words: (Optional) Anzahl von haeufigsten Woertern (top n). Falls
                n_words mit None belegt ist, sollen alle vorkommenden Woerter
                betrachtet werden.

        Returns:
            words_topn: Python Liste, die (top-n) am haeufigsten vorkommenden
                Woerter enthaelt. Die Sortierung der Liste ist nach Haeufigkeit
                absteigend.
        """
        d = defaultdict(int)
        for k in word_list:
            d[k] += 1


        sortedList = sorted(d.items(), key=itemgetter(1), reverse = True) #cmp=lambda x,y: cmp(x[1], y.[1])

        topn = sortedList[:n_words]
        return [tup[0] for tup in topn]


class WordListNormalizer(object):

    def __init__(self, stoplist=None, stemmer=None):
        """Initialisiert die Filter

        Params:
            stoplist: Python Liste von Woertern, die entfernt werden sollen
                (stopwords). Optional, verwendet NLTK stopwords falls None
            stemmer: Objekt, das die stem(word) Funktion implementiert. Optional,
                verwendet den Porter Stemmer falls None.
        """

        if stoplist is None:
            stoplist = CorpusLoader.stopwords_corpus()
        self.__stoplist = stoplist

        if stemmer is None:
            stemmer = PorterStemmer()
        self.__stemmer = stemmer
        self.__punctuation = string.punctuation
        self.__delimiters = ["''", '``', '--']


    def normalize_words(self, word_list):
        """Normalisiert die gegebenen Woerter nach in der Methode angwendeten
        Filter-Regeln (Gross-/Kleinschreibung, stopwords, Satzzeichen,
        Bindestriche und Anfuehrungszeichen, Stemming)

        Params:
            word_list: Python Liste von Worten.

        Returns:
            word_list_filtered, word_list_stemmed: Tuple von Listen
                Bei der ersten Liste wurden alle Filterregeln, bis auch stemming
                angewandt. Bei der zweiten Liste wurde zusaetzlich auch stemming
                angewandt.
        """
        stopwords = CorpusLoader.stopwords_corpus()
        punctation = string.punctuation

        filteredUpper = [w.lower() for w in word_list]
        filteredStopwords = [w for w in filteredUpper if w not in stopwords]
        word_list_filtered = [w for w in filteredStopwords if w not in punctation and w not in self.__delimiters]
        word_list_stemmed = [PorterStemmer().stem(w) for w in word_list_filtered]

        return word_list_filtered, word_list_stemmed


class IdentityFeatureTransform(object):
    """Realisert eine Transformation auf die Identitaet, bei der alle Daten
    auf sich selbst abgebildet werden. Die Klasse ist hilfreich fuer eine
    softwaretechnisch elegante Realisierung der Funktionalitaet "keine Transformation
    der Daten durchfuehren" (--> Duck-Typing).
    """
    def estimate(self, train_data, train_labels):
        pass
    def transform(self, data):
        return data


class TopicFeatureTransform(object):
    """Realsiert statistische Schaetzung eines Topic Raums und Transformation
    in diesen Topic Raum.
    """
    def __init__(self, n_topics):
        """Initialisiert die Berechnung des Topic Raums

        Params:
            n_topic: Groesse des Topic Raums, d.h. Anzahl der Dimensionen.
        """
        self.__n_topics = n_topics
        # Transformation muss in der estimate Methode definiert werden.
        self.__T = None
        self.__S_inv = None

    def estimate(self, train_data, train_labels):
        """Statistische Schaetzung des Topic Raums

        Params:
            train_data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
                Hinweis: Fuer den hier zu implementierenden Topic Raum werden die
                Klassenlabels nicht benoetigt. Sind sind Teil der Methodensignatur
                im Sinne einer konsitenten und vollstaendigen Verwaltung der zur
                Verfuegung stehenden Information.

            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')

    def transform(self, data):
        """Transformiert Daten in den Topic Raum.

        Params:
            data: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            data_trans: ndarray der in den Topic Raum transformierten Daten
                (d x n_topics).
        """
        raise NotImplementedError('Implement me')

