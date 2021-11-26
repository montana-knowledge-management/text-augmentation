"""
Example code for representing text text_augmentation with Distiller's text_augmentation tool.
"""
# from distiller.wordnet_augmentation import WordNetAugmentation
from text_augmentation.data_augmentation import WordVectorAugmenter, EasyDataAugmentation
from digital_twin_distiller.text_readers import TextReader
from importlib_resources import files
from time import time

# example text for text_augmentation
test_text = [TextReader().read(files('examples') / "data" / "augmentation_example.txt")]


def word_vector_augmentation():
    """
    Example for using DataAugmenter class.
    :return:
    """
    augmenter = WordVectorAugmenter()
    print("Loading custom fasttext model.")
    # Note that this path should point to the downloaded fasttext bin file for the required language.
    augmenter.load_fasttext_model(str(files("resources") / "cc.hu.2.bin"))
    # print("Building vocabulary.")
    # augmenter.build_vocab(test_text)
    # print("Building most similar dict.")
    # augmenter.build_most_similar_dictionary(mode="fasttext")
    augmenter.load_most_similar_dictionary(files("resources") / "most_similar_dict.json")
    print("Tokenizing.")
    tokenized_text = [test_text[0].split()]
    print("Augmenting.")
    augmenter.augment_text(tokenized_text, augmented_size=10, protected_words=["garázdaság", "vétsége", "Garázdaság"])
    for elem in augmenter.augmented_text:
        print(" ".join(elem))
    print(len(augmenter.augmented_text))
    return augmenter.augmented_text


def eda_augmentation(mode="RD", protected_words=["garázdaság", "vétsége", "Garázdaság"]):
    """
    Prints EDA augmentation for the given mode.
    :param mode: Choose from: RD: Random Deletion
                              RI: Random Insertion
                              RS: Random Swap
                              SR: Synonym Replacement
    :param protected_words: list of words that are not modified during augmentation
    :return:
    """
    eda_augmenter = EasyDataAugmentation()
    tokenized_text = [test_text[0].split()]

    eda_augmenter.augment_text(tokenized_text, 10, mode=mode, protected_words=protected_words, alpha=0.5)

    for elem in eda_augmenter.augmented_text:
        print(" ".join(elem))
    print(len(eda_augmenter.augmented_text))
    return eda_augmenter.augmented_text


if __name__ == "__main__":
    eda_augmentation()
    word_vector_augmentation()
