import numpy as np
import scipy.spatial.distance
from features import BagOfWords , IdentityFeatureTransform
from scipy.spatial.distance import cdist
from _collections import defaultdict
from operator import itemgetter




class KNNClassifier(object):
    
    def __init__(self, k_neighbors, metric):
        """Initialisiert den Klassifikator mit Meta-Parametern
        
        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string), siehe auch scipy Funktion cdist 
        """
        self.__k_neighbors = k_neighbors
        self.__metric = metric
        # Initialisierung der Membervariablen fuer Trainingsdaten als None. 
        self.__train_samples = None
        self.__train_labels = None

    def estimate(self, train_samples, train_labels):
        """Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.
        
        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden. 
        Diskutieren Sie, was den K-NN stattdessen definiert.
        
        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        
        self.__train_samples = train_samples
        self.__train_labels = train_labels
        
        
    def classify(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        if self.__train_samples is None or self.__train_labels is None:
            raise ValueError('Classifier has not been "estimated", yet!')
        train_samples =self.__train_samples
        train_labels = self.__train_labels
        
        k = self.__k_neighbors
        metrik = self.__metric
        #print "train_labels"
        #print train_labels
        distanz = cdist(test_samples, train_samples, metrik)
        #print "Distanz:"
        #print distanz

        sortiert = np.argsort(distanz, axis = 1)[:,:k]
        print "Sortirert"
        print sortiert
        
        copy_train_labels = train_labels.ravel()
        test_labels = copy_train_labels[sortiert]
        #print"test_labels"
        #print test_labels
        list_test_labels = test_labels.tolist()
        #print"list_test_labels"
        #print list_test_labels
        list_copy_train_labels = copy_train_labels.tolist()
        #print "list_copy_train_labels"
        #print list_copy_train_labels
        Bag = BagOfWords(list_copy_train_labels)
        listreturn = []
        for i in list_test_labels:
            listreturn.append(Bag.most_freq_words(i,1))
            print Bag.most_freq_words(i,1)
        #print np.asarray(listreturn)
        
        return np.asarray(listreturn)
            
        
        
        


class BayesClassifier(object):

    def __init__(self):
        """Initialisiert den Multinomial Bayes Klassifikator
        """
        # ndarray mit Klassen a-priori Wahrscheinlichkeiten
        self.__cat_apriori = None
        # ndarray mit Term-Kategorie Wahrscheinlichkeiten
        self.__term_cat_probs = None
        # ndarray mit allen bekannten Klassen (unabhaengig von Labels fuer Daten)
        self.__cat_labels = None
    
    def estimate(self, train_samples, train_labels):
        """Trainiert den Multinomial Bayes Klassfikator.
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')
    
    def classify(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        if self.__cat_apriori is None or self.__term_cat_probs is None or \
                                            self.__cat_labels is None:
            raise ValueError('BayesClassifier has not been estimated!')
        
        raise NotImplementedError('Implement me')



