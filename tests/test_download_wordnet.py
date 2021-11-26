import unittest
from text_augmentation.download_wordnet import WordNetDownloader
import os
from shutil import copyfile
from importlib_resources import files


class DownloadTestCase(unittest.TestCase):
    def test_init(self):
        wordnet_downloader = WordNetDownloader()
        self.assertEqual(wordnet_downloader.download_needed, False)
        self.assertIn("HuWN_final4.xml", str(wordnet_downloader.wordnet_xml_path))
        self.assertEqual(wordnet_downloader.url_wordnet,
                         "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/HuWN.zip")
        self.assertIn("HuWN.zip", str(wordnet_downloader.wordnet_zip_file))

    def test_check_existing_file(self):
        wordnet_downloader = WordNetDownloader()
        self.assertEqual(wordnet_downloader.download_needed, False)
        wordnet_downloader.check_existing_file()
        download_needed = not os.path.isfile(wordnet_downloader.wordnet_xml_path)
        self.assertEqual(download_needed, wordnet_downloader.download_needed)

    def test_check_existing_file_2(self):
        zip_path = str(files("tests") / "data_test" / "wordnet_example_file.zip")
        xml_path = str(files("tests") / "data_test" / "wordnet_example_file.xml")

        wordnet_downloader = WordNetDownloader(wordnet_zip_path=zip_path,
                                               wordnet_xml_path=xml_path)
        wordnet_downloader.check_existing_file()
        self.assertEqual(wordnet_downloader.download_needed, False)

    def test_unzip_cleanup(self):
        zip_path = str(files("tests") / "data_test" / "wordnet_example_file.zip")
        tmp_zip_file = "/tmp/wordnet_example_file.zip"
        tmp_xml_file = "/tmp/wordnet_example_file.xml"
        copyfile(zip_path, tmp_zip_file)
        wordnet_downloader = WordNetDownloader(wordnet_zip_path=tmp_zip_file,
                                               wordnet_xml_path=tmp_xml_file)
        self.assertTrue(os.path.isfile(wordnet_downloader.wordnet_zip_file))
        wordnet_downloader.unzip()
        self.assertTrue(os.path.isfile(files("resources") / "wordnet_example_file.xml"))
        os.remove(files("resources") / "wordnet_example_file.xml")
        wordnet_downloader.cleanup()
        self.assertFalse(os.path.isfile(wordnet_downloader.wordnet_zip_file))

    def test_run(self):
        zip_path = str(files("tests") / "data_test" / "wordnet_example_file.zip")
        tmp_zip_file = "/tmp/wordnet_example_file.zip"
        tmp_xml_file = "/tmp/wordnet_example_file.xml"
        extracted_xml = files("resources") / "wordnet_example_file.xml"
        copyfile(zip_path, tmp_zip_file)
        wordnet_downloader = WordNetDownloader(wordnet_zip_path=tmp_zip_file,
                                               wordnet_xml_path=tmp_xml_file, url_wordnet=None)
        wordnet_downloader.run()
        self.assertFalse(os.path.isfile(wordnet_downloader.wordnet_zip_file))
        self.assertTrue(os.path.isfile(extracted_xml))
        os.remove(extracted_xml)
        self.assertFalse(os.path.isfile(extracted_xml))


if __name__ == '__main__':
    unittest.main()
