"""
Example code for representing text text_augmentation with Distiller's text_augmentation tool.
"""
# from distiller.wordnet_augmentation import WordNetAugmentation
from text_augmentation.data_augmentation import WordVectorAugmenter, EasyDataAugmentation
from digital_twin_distiller.text_readers import TextReader
from importlib_resources import files

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
    print("Building vocabulary.")
    augmenter.build_vocab(test_text)
    print("Building most similar dict.")
    augmenter.build_most_similar_dictionary(mode="fasttext")
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
    from time import time

    aug_to = 2
    # word_vector_augmentation()
    # eda_augmentation()
    test_text = [
        "A horvát élvonalban játszó Kleinheisler László biztosan nem fog felépülni a magyar válogatott következő két vb selejtezőjére."]
    start_wv = time()
    augmenter = WordVectorAugmenter()
    print("Loading custom fasttext model.")
    augmenter.load_fasttext_model(
        "/media/csanyig/C8CC19CCCC19B622/Users/csanyig/Montana/NLP/Word_embeddings/cc.hu.300.bin")
    model_vw = time()
    # print("Building vocabulary.")
    # augmenter.build_vocab(test_text)
    print("Building most similar dict.")
    # augmenter.build_most_similar_dictionary(mode="fasttext")
    # augmenter.save_most_similar_dictionary("test_dict.json")
    augmenter.load_most_similar_dictionary("test_dict.json")
    print("Tokenizing.")
    tokenized_text = [test_text[0].split()]
    print("Augmenting.")
    augmenter.augment_text(tokenized_text, augmented_size=aug_to, protected_words=[], alpha=0.5)
    end_wv = time()
    for elem in augmenter.augmented_text:
        print(" ".join(elem))

    print("Fasttext: {} s, loading: {}.".format(end_wv - start_wv, model_vw - start_wv))

    protected_words = []
    start_rd = time()
    eda_augmenter = EasyDataAugmentation()
    tokenized_text = [test_text[0].split()]
    # random deletion mode
    print("EDA: RD")
    eda_augmenter.augment_text(tokenized_text, aug_to, mode="RD", protected_words=protected_words, alpha=0.5)
    end_rd = time()
    for elem in eda_augmenter.augmented_text:
        print(" ".join(elem))
    print(len(eda_augmenter.augmented_text))

    print("RD: {} s".format(end_rd - start_rd))

    start_ri = time()
    eda_augmenter = EasyDataAugmentation()
    # # random insertion
    print("EDA: RI")
    eda_augmenter.augment_text(tokenized_text, aug_to, mode="RI", protected_words=protected_words, alpha=0.5)
    end_ri = time()
    for elem in eda_augmenter.augmented_text:
        print(" ".join(elem))
    print(len(eda_augmenter.augmented_text))

    print("RI: {} s".format(end_ri - start_ri))
    # # random swap
    start_rs = time()
    eda_augmenter = EasyDataAugmentation()
    print("EDA: RS")
    eda_augmenter.augment_text(tokenized_text, aug_to, mode="RS", protected_words=protected_words, alpha=0.5)
    end_rs = time()
    for elem in eda_augmenter.augmented_text:
        print(" ".join(elem))

    print("RS: {} s".format(end_rs - start_rs))
    # print(len(eda_augmenter.augmented_text))
    # # synonym replacement
    start_sr = time()
    eda_augmenter = EasyDataAugmentation()
    print("EDA: SR")
    eda_augmenter.augment_text(tokenized_text, aug_to, mode="SR", protected_words=protected_words, alpha=0.5)
    end_sr = time()
    for elem in eda_augmenter.augmented_text:
        print(" ".join(elem))
    print(len(eda_augmenter.augmented_text))

    print("SR: {} s".format(end_sr - start_sr))
