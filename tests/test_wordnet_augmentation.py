import unittest
from digital_twin_distiller.text_readers import TextReader
from importlib_resources import files
from text_augmentation.wordnet_augmentation import WordNetAugmentation


class WordnetTestCase(unittest.TestCase):
    def test_wordnet_example(self):
        test_text = TextReader().read(
            str(files("examples") / "data" / "augmentation_example_2.txt"))
        wordnet_augmenter = WordNetAugmentation()
        wordnet_augmenter.wordnet_path = files("tests") / "data_test" / "wordnet_example_file.xml"
        wordnet_augmenter.load_synonyms_dict()
        result = wordnet_augmenter.run(test_text.split(), n=2)
        result = " ".join(result)
        self.assertFalse(test_text == result)


if __name__ == '__main__':
    unittest.main()
