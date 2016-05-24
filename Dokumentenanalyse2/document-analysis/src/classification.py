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
        #test_samples = self.__test_samples 
        k = self.__k_neighbors
        metrik = self.__metric
        distanz = cdist(test_samples, train_samples, metrik)
        sortiert = np.argsort(distanz, axis = 1)[:,:k]
        train_labels = train_labels.reshape(-1)
        print sortiert
        knachbarn = train_labels[distanz]
        
        #knachbarnlist= np.array([train_labels[np.argpartition(a, k)[:k]].flatten()] for a in distanz)
        
         returnlist = []
#         for i in knachbarn:
#             d = defaultdict(lambda: 1)
#             for j in i:
#                d[j] =+1
#             sortedList=sorted(d, key=itemgetter(1), reverse=True) 
#             returnlist.append(sortedList[:1])
#             
#         print returnlist
         return returnlist
            
        
        
        


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



