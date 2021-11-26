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

    def test_set_domain_names(self):
        wordnet_augmenter = WordNetAugmentation()
        example_set = ["this", "that"]
        wordnet_augmenter.set_wordnet_domains(example_set)
        self.assertEqual(wordnet_augmenter.wordnet_domains, example_set)

    def test_load_xml(self):
        wordnet_augmenter = WordNetAugmentation()
        with self.assertRaises(FileNotFoundError) as context:
            wordnet_augmenter.load_synonyms_dict()
            self.assertIn("HuWN_final4.xml is missing from resources, please run download_wordnet.py to download it!",
                          context.exception)

    def test_returning_original_doc(self):
        wordnet_augmenter = WordNetAugmentation()
        wordnet_augmenter.synonyms_set = set()
        intersection, can_augment = wordnet_augmenter._get_potential_words([], [], 2)
        self.assertEqual(can_augment, False)
        self.assertEqual(intersection, [])

    def test_synonym_insertion_replacement_nothing_input(self):
        wordnet_augmenter = WordNetAugmentation()
        wordnet_augmenter.synonyms_set = set()
        words = ["a"]
        # synonym insertion
        ret_val = wordnet_augmenter.synonym_insertion(words=words, protected_words=["b"])
        self.assertEqual(ret_val, words)
        # synonym replacement
        ret_val = wordnet_augmenter.synonym_replacement(words=words, protected_words=["b"])
        self.assertEqual(ret_val, words)


if __name__ == '__main__':
    unittest.main()
