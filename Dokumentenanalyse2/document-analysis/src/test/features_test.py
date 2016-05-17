import unittest
from features import WordListNormalizer, BagOfWords, RelativeTermFrequencies, \
    RelativeInverseDocumentWordFrequecies, TopicFeatureTransform
import numpy as np

class WordListNormalizerTest(unittest.TestCase):


    def setUp(self):
        self.__word_list = ['This', 'is', 'a', 'test', 'text', '.', 'It', 'was',
                            'written', 'to', 'demonstrate', 'filtering', 'and',
                            'and', '"', 'stemming', "''", '--', 'For', 'Real',
                            '!']


    def test_normalize_words(self):
        # Default Initialisierung verwendet NLTK Standard Stoplist 
        # und Porter Stemmer 
        normalizer = WordListNormalizer()
        result = normalizer.normalize_words(self.__word_list)
        word_list_filtered_ref = ['test', 'text', 'written', 'demonstrate',
                                  'filtering', 'stemming', 'real']
        word_list_stemmed_ref = ['test', 'text', 'written', 'demonstr',
                                 'filter', 'stem', 'real']
        self.assertEqual(result[0], word_list_filtered_ref)
        self.assertEqual(result[1], word_list_stemmed_ref)


class BagOfWordsTest(unittest.TestCase):

    @staticmethod
    def word_list():
        return ['test', 'text', 'written', 'demonstr',
                'filter', 'stem', 'real', 'test', 'written',
                'filter', 'real', 'test', 'demonstr', 'real']
    
    @staticmethod
    def vocabulary():
        return ['real', 'test', 'filter', 'written',
                'demonstr', 'text', 'stem' ]
    
    @staticmethod
    def cat_wordlist_dict():
        word_list = BagOfWordsTest.word_list()
        cat_word_dict = {'A' : [word_list[0:5],
                                word_list[2:7],
                                word_list[4:9]],
                         'B' : [word_list[5:11],
                                word_list[7:13],
                                ],
                         'C' : [word_list[3:10],
                                word_list[6:14],
                                ],
                         'D' : [word_list[1:8]],
                         }
        return cat_word_dict

    def setUp(self):
        self.__word_list = BagOfWordsTest.word_list()
        self.__word_list_freq_ref = BagOfWordsTest.vocabulary()
    
    def test_most_freq_words(self):
        result = BagOfWords.most_freq_words(self.__word_list)
        
        self.assertEqual(set(result), set(self.__word_list_freq_ref))
        self.assertEqual(len(result), len(self.__word_list_freq_ref))
        self.assertTrue(result.index('test') < result.index('written'))
        self.assertTrue(result.index('written') < result.index('stem'))
        self.assertTrue(result.index('real') < result.index('demonstr'))
        
        for idx in range(len(self.__word_list_freq_ref)):
            result = BagOfWords.most_freq_words(self.__word_list, n_words=idx)
            self.assertEqual(set(result), set(self.__word_list_freq_ref[:idx]))
            
    def test_category_bow_dict(self):
        cat_word_dict = BagOfWordsTest.cat_wordlist_dict()
        # Verwendet absolute Term Frequenzen als Default
        bow = BagOfWords(vocabulary=self.__word_list_freq_ref)
        cat_bow_dict = bow.category_bow_dict(cat_word_dict)

        A_bow_ref = np.array([[ 0., 1., 1., 1., 1., 1., 0.],
                              [ 1., 0., 1., 1., 1., 0., 1.],
                              [ 1., 1., 1., 1., 0., 0., 1.]])
        B_bow_ref = np.array([[ 2., 1., 1., 1., 0., 0., 1.],
                              [ 1., 2., 1., 1., 1., 0., 0.]])
        C_bow_ref = np.array([[ 1., 1., 2., 1., 1., 0., 1.],
                              [ 3., 2., 1., 1., 1., 0., 0.]])
        D_bow_ref = np.array([[ 1., 1., 1., 1., 1., 1., 1.]])
        np.testing.assert_equal(cat_bow_dict['A'], A_bow_ref)
        np.testing.assert_equal(cat_bow_dict['B'], B_bow_ref)
        np.testing.assert_equal(cat_bow_dict['C'], C_bow_ref)
        np.testing.assert_equal(cat_bow_dict['D'], D_bow_ref)
    
    
class TermFrequenciesTest(unittest.TestCase):
    
    def setUp(self):
        self.__bow_mat_long = np.array([[ 1., 0., 2., 5., 1., 0., 1.],
                                        [ 3., 2., 0., 2., 1., 0., 0.]])
        self.__bow_mat_short = np.array([[ 1., 6., 0., 1., 0., 2., 0.]])
    def test_relative_term_frequencies(self):
        term_freq = RelativeTermFrequencies()
        bow_weighted_long = term_freq.weighting(self.__bow_mat_long)
        bow_weighted_short = term_freq.weighting(self.__bow_mat_short)
        bow_weighted_long_ref = np.array([[ 0.1, 0., 0.2, 0.5, 0.1, 0., 0.1  ],
                                          [ 0.375, 0.25, 0., 0.25, 0.125, 0., 0. ]])     
        np.testing.assert_equal(bow_weighted_long, bow_weighted_long_ref)

        bow_weighted_short_ref = np.array([[ 0.1, 0.6, 0., 0.1, 0., 0.2, 0. ]])     
        np.testing.assert_equal(bow_weighted_short, bow_weighted_short_ref)
    
    def test_inverse_document_frequencies(self):
        vocabulary = BagOfWordsTest.vocabulary()
        cat_wordlist_dict = BagOfWordsTest.cat_wordlist_dict()
        tfidf = RelativeInverseDocumentWordFrequecies(vocabulary,
                                                      cat_wordlist_dict)
        self.assertEqual(self.__bow_mat_short.shape[1], len(vocabulary))
        bow_weighted_short = tfidf.weighting(self.__bow_mat_short)
        bow_weighted_short_ref = np.array([[ 0.01335314, 0.08011884, 0., 0., 0.,
                                             0.27725887, 0.]])
        np.testing.assert_almost_equal(bow_weighted_short, bow_weighted_short_ref)

        self.assertEqual(self.__bow_mat_long.shape[1], len(vocabulary))
        bow_weighted_long = tfidf.weighting(self.__bow_mat_long)
        bow_weighted_long_ref = np.array([[ 0.01335314, 0., 0., 0., 0.02876821,
                                            0., 0.04700036],
                                          [ 0.05007427, 0.03338285, 0., 0.,
                                           0.03596026, 0., 0.]])
        np.testing.assert_almost_equal(bow_weighted_long, bow_weighted_long_ref)

class TopicFeatureTransformTest(unittest.TestCase):

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

    def test_estimate(self):
        T_mat_ref = np.array([[-0.41334067, 0.8048751, 0.42582339],
                              [-0.76349026, -0.05150169, -0.64376254],
                              [-0.49621781, -0.59120525, 0.63580204]])
        S_inv_mat_ref = np.array([[ 0.04730404, 0., 0.],
                                  [ 0., 0.10312641, 0.],
                                  [ 0., 0., 0.25752625]])
        feature_trans = TopicFeatureTransform(n_topics=3)
        feature_trans.estimate(self.__train_samples, train_labels=None)
        # Achtung: Zugriff auf private Variablen
        # Wenn das Modell ausserhalb der Klasse benoetigt wird sollte ein
        # regulaerer Zugriff moeglich sein.
        T_mat = feature_trans._TopicFeatureTransform__T  # IGNORE:protected-access
        S_inv_mat = feature_trans._TopicFeatureTransform__S_inv  # IGNORE:protected-access
        np.testing.assert_array_almost_equal(T_mat, T_mat_ref)
        np.testing.assert_array_almost_equal(S_inv_mat, S_inv_mat_ref)
        
        feature_trans = TopicFeatureTransform(n_topics=2)
        feature_trans.estimate(self.__train_samples, train_labels=None)
        T_mat = feature_trans._TopicFeatureTransform__T  # IGNORE:protected-access
        S_inv_mat = feature_trans._TopicFeatureTransform__S_inv  # IGNORE:protected-access
        np.testing.assert_array_almost_equal(T_mat, T_mat_ref[:, :2])
        np.testing.assert_array_almost_equal(S_inv_mat, S_inv_mat_ref[:2, :2])
    
    def test_transform(self):
        feature_trans = TopicFeatureTransform(n_topics=2)
        feature_trans.estimate(self.__train_samples, train_labels=None)
        test_samples_trans = feature_trans.transform(self.__test_samples)
        test_samples_trans_ref = np.array([[-0.09776343, 0.41501939],
                                           [-0.18058089, -0.02655592],
                                           [-0.11736555, -0.30484437],
                                           [-0.27834432, 0.38846347],
                                           [-0.29794644, -0.33140029]])
        np.testing.assert_array_almost_equal(test_samples_trans, 
                                             test_samples_trans_ref)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
