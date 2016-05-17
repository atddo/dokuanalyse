import unittest
from classification import KNNClassifier, BayesClassifier
import numpy as np

class ClassificationTest(unittest.TestCase):

    

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.__train_samples = np.array([[2, 5, 0],
                                         [4, 1, 0],
                                         [3, 3, 1],
                                         [9, 8, 2],
                                         [1, 5, 3],
                                         [0, 7, 9],
                                         [2, 9, 6],
                                         [0, 2, 3],
                                         [5, 3, 3]]) 
        self.__test_samples = np.array([[5, 0, 0],
                                        [0, 5, 0],
                                        [0, 0, 5],
                                        [5, 5, 0],
                                        [0, 5, 5]])
        self.__train_labels = np.array([['A', 'A', 'C', 'A', 'C', 'B', 'B', 'C', 'C' ]]).T

    def test_knn(self):
        print 'knn_test'
        knn = KNNClassifier(k_neighbors=1, metric='cityblock')
        knn.estimate(self.__train_samples, self.__train_labels)
        result_labels = knn.classify(self.__test_samples)
        result_labels_ref = np.array([['A', 'A', 'C', 'A', 'C' ]]).T
        self.assertEqual(result_labels_ref.shape, result_labels.shape)
        self.assertEqual(result_labels_ref.dtype, result_labels.dtype)
        np.testing.assert_equal(result_labels, result_labels_ref)
        
        knn = KNNClassifier(k_neighbors=3, metric='cityblock')
        knn.estimate(self.__train_samples, self.__train_labels)
        result_labels = knn.classify(self.__test_samples)
        result_labels_ref = np.array([['C', 'C', 'C', 'A', 'C' ]]).T
        np.testing.assert_equal(result_labels, result_labels_ref)

    def test_bayes(self):
        print 'bayes_test'
        bayes = BayesClassifier()
        bayes.estimate(self.__train_samples, self.__train_labels)
        result_labels = bayes.classify(self.__test_samples)
        result_labels_ref = np.array([['A', 'A', 'B', 'A', 'B' ]]).T
        self.assertEqual(result_labels_ref.shape, result_labels.shape)
        self.assertEqual(result_labels_ref.dtype, result_labels.dtype)
        np.testing.assert_equal(result_labels, result_labels_ref)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
