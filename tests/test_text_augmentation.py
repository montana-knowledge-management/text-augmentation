import unittest
from examples.text_augmentation_example import eda_augmentation, word_vector_augmentation


class AugmentationTest(unittest.TestCase):
    def test_eda(self):
        result = eda_augmentation()
        self.assertEqual(len(result), 10)

    def test_wv(self):
        result = word_vector_augmentation()
        self.assertEqual(len(result), 10)
