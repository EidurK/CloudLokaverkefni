
import sys
import os

sys.path.append(os.path.abspath("../src/lib"))

import unittest
import clean_data as dta

class TestDataProcessing(unittest.TestCase):
    test_row = 'Hello!  I need to https://www.youtube.com/ show link https://www.youtube.com/'
    def test_remove_links(self):
        result = dta.remove_links(self.test_row)
        expected = 'Hello! I need to show link'
        self.assertEqual(result, expected)

    def test_remove_symbols(self):
        result = dta.remove_symbols(self.test_row)
        expected = 'hello i need to httpswwwyoutubecom show link httpswwwyoutubecom'

        self.assertEqual(result, expected)

    def test_stopwords_lemmatizer(self):
        test_text = "this wouldn't work BY appending the trump"
        stopwords = ['you', 'this', 'by', 'the', "wouldn't"]
        result = dta.stopwords_lemmatizer(test_text, stopwords, testing=True)
        expected = "work append trump"
        self.assertEqual(result, expected)


    def test_clean_row(self):
        stopwords = []
        result = dta.clean_row(self.test_row, stopwords)
        expected = 'hello i need to show link'
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
