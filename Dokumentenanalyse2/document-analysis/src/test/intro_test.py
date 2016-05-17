import unittest
import types
import numpy as np
from collections import defaultdict
from intro.main import PythonIntro, NumPyIntro


class PythonIntroTest(unittest.TestCase):

    def setUp(self):
        self.__py_intro = PythonIntro()

    def test_datatypes(self):
        var_tup = (1, 2, 3)
        self.assertFalse(self.__py_intro.datatypes(var_tup))
        var_tup = (1, 2, 3, 23)
        self.assertTrue(self.__py_intro.datatypes(var_tup))
        var_tup = (1, 2, 3, 42)
        self.assertTrue(self.__py_intro.datatypes(var_tup))
        var_tup = (1, 2, 3, 23, 42)
        self.assertTrue(self.__py_intro.datatypes(var_tup))

    def test_sequences(self):
        result = self.__py_intro.sequences(seq_start=10, seq_end=33, seq_step=2)
        # Check result
        seq_list = result[0] 
        # Laenge der Liste
        self.assertEqual(seq_list[0], 12)
        # erstes Element
        self.assertEqual(seq_list[1], 10)
        # letztes Element
        self.assertEqual(seq_list[2], 32)
        # letztes und vorletztes Element
        self.assertEqual(seq_list[3], [30, 32])
        # erstes Drittel
        self.assertEqual(seq_list[4], range(10, 18, 2))
        # zweites Drittel
        self.assertEqual(seq_list[5], range(18, 26, 2))
        # drittes Drittel
        self.assertEqual(seq_list[6], range(26, 34, 2))
        
        str_list = result[1]
        str_list_ref = ['0: 10 ', '1: 12 ', '2: 14 ', '3: 16 ', '4: 18 ',
                        '5: 20 ', '6: 22 ', '7: 24 ', '8: 26 ', '9: 28 ',
                        '10: 30 ', '11: 32 ']
        self.assertEqual(str_list, str_list_ref)
        
    def test_sequences_complex(self):
        result = self.__py_intro.sequences_complex(test_seq=[True,
                                                             (1, 2),
                                                             {'key':'value'}])
        str_ref = "[ <bool, True>, <tuple, (1, 2)>, <dict, {'key': 'value'}> ]"
        self.assertEqual(result, str_ref)
        
    def test_list_comprehension(self):
        result = self.__py_intro.list_comprehension(test_seq=[True,
                                                    (1, 2),
                                                    {'key':'value'}])
        self.assertEqual(type(result), types.GeneratorType)
        result_list = list(result)
        list_ref = ['<bool, True>', '<tuple, (1, 2)>',
                    "<dict, {'key': 'value'}>"]
        self.assertEqual(result_list, list_ref)
        
    def test_dictionaries(self):
        sample_data_tup = (np.arange(1, 11) + 0.2,
                           np.arange(4, 15) + 0.7,
                           np.arange(5, 11))
        sample_arr = np.hstack(sample_data_tup)
        sample_list = list(sample_arr) 
        
        result = self.__py_intro.dictionaries(rand_list=sample_list)
        hist = result[0]
        self.assertEqual(type(hist), defaultdict)
        self.__check_hist(hist, seq_range=(1, 5), expected=1)
        self.__check_hist(hist, seq_range=(5, 11), expected=3)
        self.__check_hist(hist, seq_range=(11, 16), expected=1)
            
        max_key = result[1]
        self.assertTrue(max_key in range(5, 11))
        
    def __check_hist(self, hist, seq_range, expected):
        for val in range(*seq_range):
            self.assertEqual(hist[val], expected)
            
            
class NumPyIntroTest(unittest.TestCase):

    def setUp(self):
        self.__np_intro = NumPyIntro()
        self.__test_arr = np.array([[199., 11., 192., 179., 93., 9., 170.],
                                    [192., 165., 41., 113., 24., 73., 131.],
                                    [ 94., 154., 53., 133., 136., 190., 150.],
                                    [  4., 177., 25., 85., 68., 72., 105.]])

    def test_arrays(self):
        result = self.__np_intro.arrays(test_arr=self.__test_arr)
        arr = result[0]
        arr_ref = np.array([[ 5, 6, 7],
                            [ 8, 9, 10],
                            [11, 12, 13],
                            [14, 15, 16]])
        np.testing.assert_equal(arr, arr_ref)
        arr_lin_rows = result[1]
        arr_lin_rows_ref = np.array([ 199., 11., 192., 179., 93., 9., 170.,
                                     192., 165., 41., 113., 24., 73., 131.,
                                     94., 154., 53., 133., 136., 190., 150.,
                                     4., 177., 25., 85., 68., 72., 105.])
        np.testing.assert_equal(arr_lin_rows, arr_lin_rows_ref)
        arr_lin_cols = result[2]
        arr_lin_cols_ref = np.array([ 199., 192., 94., 4., 11., 165., 154.,
                                     177., 192., 41., 53., 25., 179., 113.,
                                     133., 85., 93., 24., 136., 68., 9.,
                                     73., 190., 72., 170., 131., 150., 105.])
        np.testing.assert_equal(arr_lin_cols, arr_lin_cols_ref)
        
    def test_array_access(self):
        result = self.__np_intro.array_access(self.__test_arr)
        self.assertEqual(result, 136)
        
    def test_array_slicing(self):
        result = self.__np_intro.array_slicing(self.__test_arr)
        sub_arr0_ref = np.array([[199., 11.],
                                 [192., 165.],
                                 [94., 154.]])
        np.testing.assert_array_equal(result[0], sub_arr0_ref)
        
        sub_arr1_ref = np.array([[ 24., 73., 131.],
                                 [ 136., 190., 150.],
                                 [ 68., 72., 105.]])
        np.testing.assert_array_equal(result[1], sub_arr1_ref)
        
        sub_arr2_ref = np.array([  4., 177., 25., 85., 68., 72., 105.])
        np.testing.assert_array_equal(result[2], sub_arr2_ref)
        
        sub_arr3_ref = np.array([[  11., 179., 9.],
                                 [ 165., 113., 73.],
                                 [ 154., 133., 190.],
                                 [ 177., 85., 72.]])
        np.testing.assert_array_equal(result[3], sub_arr3_ref)
        
    def test_array_indexing(self):
        result = self.__np_intro.array_indexing(self.__test_arr)
        sub_arr0_ref = np.array([ 113., 154., 177.])
        np.testing.assert_array_equal(result[0], sub_arr0_ref)
        
        sub_arr1_ref = np.array([[ 192., 179., 170.],
                                 [  41., 113., 131.],
                                 [  53., 133., 150.],
                                 [  25., 85., 105.]])
        np.testing.assert_array_equal(result[1], sub_arr1_ref)
        
        sub_arr2_ref = np.array([[   0., 0., 192., 0., 0., 0., 170.],
                                 [ 192., 0., 0., 0., 24., 0., 0.],
                                 [  94., 154., 0., 0., 136., 190., 150.],
                                 [   4., 0., 0., 0., 68., 72., 0.]])
        np.testing.assert_array_equal(result[2], sub_arr2_ref)
        
        sub_arr3_ref = np.array([ 192., 192., 24., 150., 72.])
        np.testing.assert_array_equal(result[3], sub_arr3_ref)
        
    def test_array_operations(self):
        result = self.__np_intro.array_operations(self.__test_arr)
        sub_arr0_ref = np.array([ 391., 176., 233., 292., 117., 82., 301.])
        np.testing.assert_array_equal(result[0], sub_arr0_ref)
        
        sub_arr1_ref = np.array([ 38208., 1815., 7872., 20227.,
                                 2232., 657., 22270.])
        np.testing.assert_array_equal(result[1], sub_arr1_ref)
        
        self.assertEqual(result[2], 93281.0)
        
    def test_array_functions(self):
        result = self.__np_intro.array_functions(self.__test_arr)
        
        sub_arr0_ref = (np.array([0, 0, 5, 1]),
                        np.array([ 199., 192., 190., 177.]))
        np.testing.assert_array_equal(result[0], sub_arr0_ref)
        
        sub_arr1_ref = np.array([ 853., 739., 910., 536.])
        np.testing.assert_array_equal(result[1], sub_arr1_ref)

        sub_arr2_ref = np.array([[ 0.23329426, 0.01289566, 0.22508792,
                                  0.2098476 , 0.10902696, 0.010551  , 0.1992966 ],
                                 [ 0.25981055, 0.2232747 , 0.05548038,
                                  0.15290934, 0.03247632, 0.09878214, 0.17726658],
                                 [ 0.1032967 , 0.16923077, 0.05824176,
                                  0.14615385, 0.14945055, 0.20879121, 0.16483516],
                                 [ 0.00746269, 0.33022388, 0.04664179,
                                  0.15858209, 0.12686567, 0.13432836, 0.19589552]])
        np.testing.assert_array_almost_equal(result[2], sub_arr2_ref)
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
