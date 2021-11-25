import unittest
from digital_twin_distiller.text_readers import TextReader
from importlib_resources import files
from text_augmentation.wordnet_augmentation import WordNetAugmentation


class WordnetTestCase(unittest.TestCase):
    def test_wordnet_example(self):
        test_text = TextReader().read(
            str(files("examples") / "data" / "augmentation_example_2.txt"))
        print(test_text)
        wordnet_augmenter = WordNetAugmentation()
        wordnet_augmenter.load_synonyms_dict()
        result = " ".join(wordnet_augmenter.run(test_text.split()))
        self.assertFalse(test_text == result)


if __name__ == '__main__':
    unittest.main()
