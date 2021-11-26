import unittest

import numpy as np
from text_augmentation.data_augmentation import WordVectorAugmenter
from text_augmentation.data_augmentation import EasyDataAugmentation
from importlib_resources import files
from scipy.stats import entropy
import os

example_input = [
    "KÚRIA Bfv szám Kúria Budapesten július napján ezért megtartott tanácsülésen meghozta következő garázdaság vétsége volt".split(),
    "miatt folyamatban volt büntetőügyben Legfőbb Ügyészség ezért által előterjesztett felülvizsgálati indítványt".split(),
]


class DummyFasttextModel:
    def __init__(self):
        self.inner_vocab = {"megtartott": [[0.985296392440796, "megtart"], [0.6585092544555664, "megrendezett"]],
                            "indítványt": [[0.8903484106063843, "indítványokat"], [0.7796140909194946, "indítványát"]]}

    def get_nearest_neighbors(self, word: str, k=2):
        return self.inner_vocab.get(word)


# for word in self.vocabulary:
#     if word in self.gensim_model.wv.key_to_index:
#         if word not in self.most_similar_dict:
#             self.most_similar_dict[word] = self.gensim_model.wv.most_similar(
#                 word, topn=self.topn_most_similar
#             )
#     else:

class WordVectorDummy:
    def __init__(self):
        self.key_to_index = ["megtartott", "indítványt"]

        self.inner_vocab = {"megtartott": [("megtart", 0.985296392440796), ("megrendezett", 0.6585092544555664)],
                            "indítványt": [("indítványokat", 0.8903484106063843), ("indítványát", 0.7796140909194946)]}

    def most_similar(self, word, topn=2):
        return self.inner_vocab.get(word)


class DummyGensimModel:
    def __init__(self):
        self.wv = WordVectorDummy()


class AugmentationTestCase(unittest.TestCase):
    def test_first_pick(self):
        lst = ["a", "b", "c"]
        d = WordVectorAugmenter()
        pick = d.first_pick(lst)
        self.assertEqual(pick, "a")

    def test_random_and_wighted_pick(self):
        lst = [("a", 3), ("b", 2), ("c", 1)]
        d = WordVectorAugmenter()
        random_picks = {"a": 0, "b": 0, "c": 0}
        probability_weighted_picks = {"a": 0, "b": 0, "c": 0}
        for _ in range(10000):
            pick = d.random_pick(lst)
            random_picks[pick] += 1
            pick = d.weighted_probability_pick(lst)
            probability_weighted_picks[pick] += 1
        # entropy of uniformly distributed picking must be higher than the weighted distribution
        entropy_random = entropy(np.array(list(random_picks.values())) / np.int64(1000), base=2)
        entropy_weighted = entropy(np.array(list(probability_weighted_picks.values())) / np.int64(1000), base=2)
        self.assertGreater(entropy_random, entropy_weighted)

    def test_load_most_similar_words(self):
        d = WordVectorAugmenter()
        d.load_most_similar_dictionary(files("resources") / "most_similar_dict.json")
        self.assertIsNotNone(d.most_similar_dict)
        self.assertEqual(d.most_similar_dict["KÚRIA"][0][0], "KÚRIAI")

    def test_save_most_similar_words(self):
        d = WordVectorAugmenter()
        dict_path = "/tmp/exmaple.json"
        d.most_similar_dict = {"a": "b"}
        d.save_most_similar_dictionary(dict_path)
        self.assertTrue(os.path.isfile(dict_path))
        d.most_similar_dict = {"c": "d"}
        d.load_most_similar_dictionary(dict_path)
        self.assertDictEqual(d.most_similar_dict, {"a": "b"})

    def test_augment_text_error(self):
        d = WordVectorAugmenter()
        with self.assertRaises(ValueError) as _:
            d.augment_text(example_input, 2)

    def test_augment_text(self):
        d = WordVectorAugmenter()
        augment_to = 10
        d.load_most_similar_dictionary(files("resources") / "most_similar_dict.json")
        d.augment_text(example_input, augment_to)
        self.assertEqual(len(d.augmented_text), augment_to)

    def test_augment_text_(self):
        d = WordVectorAugmenter()
        augment_to = 10
        d.load_most_similar_dictionary(files("resources") / "most_similar_dict.json")
        d.augment_text(example_input, augment_to, picking_mode="random")
        self.assertEqual(len(d.augmented_text), augment_to)

    # def test_wordnet_augmentation(self):
    #     d = WordVectorAugmenter()
    #     augment_to = 10
    #     d.augment_text(example_input, augment_to, picking_mode="wordnet")
    #     self.assertEqual(len(d.augmented_text), augment_to)

    def test_eda_deletion(self):
        eda = EasyDataAugmentation()
        sample = ["ez", "egy", "minta"]
        eda.set_protected_words(["ez"])
        self.assertEqual(eda.random_deletion(sample, 1), ["ez"])
        self.assertEqual(eda.random_deletion(sample, 0), ["ez", "egy", "minta"])

    def test_eda_swap(self):
        eda = EasyDataAugmentation()
        sample = ["ez", "egy", "minta"]
        self.assertIsNot(eda.random_swap(sample, 1), sample)

    def test_eda_insert(self):
        eda = EasyDataAugmentation()
        eda.init_wordnet(files("tests") / "data_test" / "wordnet_example_file.xml")
        sample = ["ez", "egy", "piros", "mégis"]
        # since one synonym is inserted, the length should be one more than the original data
        new_data = eda.random_insertion(sample, 1)
        self.assertGreater(len(new_data), len(sample))

    def test_eda_synonym_replacement(self):
        eda = EasyDataAugmentation()
        eda.init_wordnet(files("tests") / "data_test" / "wordnet_example_file.xml")
        sample = ["ez", "egy", "minta", "tulajdonképp", "mégis"]
        # since only one word is replaced by its synonym, the intersection with the original data is one less than the
        # length of the original data
        intersection = set(eda.synonym_replacement(sample, 1)).intersection(set(sample))
        self.assertEqual(len(intersection), len(sample) - 1)

    def test_build_vocab(self):
        d = WordVectorAugmenter()
        text = [" ".join(example) for example in example_input]
        params = {"lowercase": True}
        d.build_vocab(text)
        self.assertEqual(len(d.vocabulary), 24)
        d.build_vocab(text, **params)
        self.assertEqual(len(d.vocabulary), 23)

    def test_build_most_similar_dict_fasttext(self):
        d = WordVectorAugmenter()
        text = ["megtartott", "indítványt"]
        d.fasttext_model = DummyFasttextModel()
        d.build_vocab(text)
        d.build_most_similar_dictionary(mode="fasttext")
        self.assertEqual(d.most_similar_dict.get("megtartott")[0], ('megtart', 0.985296392440796))
        self.assertEqual(d.most_similar_dict.get("indítványt")[0], ('indítványokat', 0.8903484106063843))

    def test_build_most_similar_dict_gensim(self):
        d = WordVectorAugmenter()
        text = ["megtartott", "indítványt"]
        d.gensim_model = DummyGensimModel()
        d.build_vocab(text)
        d.build_most_similar_dictionary(mode="gensim")
        self.assertEqual(d.most_similar_dict.get("megtartott")[0], ('megtart', 0.985296392440796))
        self.assertEqual(d.most_similar_dict.get("indítványt")[0], ('indítványokat', 0.8903484106063843))

    def test_build_most_similar_dict_error(self):
        d = WordVectorAugmenter()
        with self.assertRaises(ValueError) as context:
            d.build_most_similar_dictionary(mode="fasttext")
        with self.assertRaises(ValueError) as context:
            d.build_most_similar_dictionary(mode="gensim")
            self.assertIn("Missing loaded gensim model. Please load gensim model!", context.exception)
        with self.assertRaises(ValueError) as context:
            d.build_most_similar_dictionary(mode="not defined")
            self.assertIn("Wrong mode given! Please choose from 'gensim' or 'fasttext'!", context.exception)

    def test_augment_main(self):
        d = WordVectorAugmenter()
        protected_lst = ["garázdaság", "vétsége", "Legfőbb", "Ügyészség"]
        text = [" ".join(example) for example in example_input]
        d.build_vocab(text)
        # loading previously calculated similarities
        d.load_most_similar_dictionary(files("resources") / "most_similar_dict.json")
        d.augment_text(example_input, 10, protected_words=protected_lst, alpha=1.0)
        # protected words must remain intact
        for word in protected_lst:
            word_count = 0
            for doc in d.augmented_text:
                word_count += doc.count(word)
            self.assertEqual(word_count, 5)

        # not protected words must be changed
        july_count = 0
        for doc in d.augmented_text:
            july_count += doc.count("július")
        self.assertNotEqual(july_count, 5)
        print(d.augmented_text)

    def test_protected_list(self):
        d = WordVectorAugmenter()
        protected_lst = ["ez", "meg", "az"]
        d.set_protected_words(protected_lst)
        self.assertListEqual(d.protected_words, protected_lst)

    def test_eda_augment_text_random_deletion(self):
        eda = EasyDataAugmentation()
        eda.init_wordnet(files("tests") / "data_test" / "wordnet_example_file.xml")
        eda.augment_text(example_input, 10, mode="RD", protected_words=["garázdaság", "volt"], alpha=0.5)
        max_size = max(len(example_input[0]), len(example_input[1]))
        for elem in eda.augmented_text[2:]:
            # checking if the word volt can be found in all documents
            self.assertIn("volt", elem)
            self.assertLess(len(elem), max_size)

    def test_eda_augment_text_random_insertion(self):
        eda = EasyDataAugmentation()
        eda.init_wordnet(files("tests") / "data_test" / "wordnet_example_file.xml")
        eda.augment_text(example_input, 10, mode="RI", alpha=0.5, protected_words=["volt"])
        for idx, elem in enumerate(eda.augmented_text[2:]):
            # checking if the word volt can be found in all documents
            self.assertIn("volt", elem)
            if idx % 2 == 0:
                # after insertion the length should be higher
                self.assertGreater(len(elem), len(example_input[0]))
                self.assertNotEqual(" ".join(elem), " ".join(example_input[0]))
            if idx % 2 == 1:
                # after insertion the length should be higher
                self.assertGreater(len(elem), len(example_input[1]))
                self.assertNotEqual(" ".join(elem), " ".join(example_input[1]))

    def test_eda_augment_text_random_swap(self):
        eda = EasyDataAugmentation()
        eda.init_wordnet(files("tests") / "data_test" / "wordnet_example_file.xml")
        eda.augment_text(example_input, 10, mode="RS", alpha=0.2)
        for idx, elem in enumerate(eda.augmented_text[2:]):
            if idx % 2 == 0:
                # same length as original document
                self.assertEqual(len(elem), len(example_input[0]))
                # after swapping the order is not the same
                self.assertNotEqual(" ".join(elem), " ".join(example_input[0]))
            if idx % 2 == 1:
                # same length as original document
                self.assertEqual(len(elem), len(example_input[1]))
                # after swapping the order is not the same
                self.assertNotEqual(" ".join(elem), " ".join(example_input[1]))

    def test_eda_augment_text_synonym_replacement(self):
        eda = EasyDataAugmentation()
        eda.init_wordnet(files("tests") / "data_test" / "wordnet_example_file.xml")
        eda.augment_text(example_input, 10, mode="SR", alpha=0.5, protected_words=["volt"])
        for idx, elem in enumerate(eda.augmented_text[2:]):
            # checking if the word volt can be found in all documents
            self.assertIn("volt", elem)
            if idx % 2 == 0:
                self.assertEqual(len(elem), len(example_input[0]))
                self.assertNotEqual(" ".join(elem), " ".join(example_input[0]))
            if idx % 2 == 1:
                self.assertEqual(len(elem), len(example_input[1]))
                self.assertNotEqual(" ".join(elem), " ".join(example_input[1]))


if __name__ == '__main__':
    unittest.main()
