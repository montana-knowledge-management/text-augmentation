import unittest
from text_augmentation.download_hungarian_wordnet import WordNetDownloader
import os

class DownloadTestCase(unittest.TestCase):
    def test_init(self):
        wordnet_downloader = WordNetDownloader()
        self.assertEqual(wordnet_downloader.download_needed, False)
        self.assertIn("HuWN_final4.xml", str(wordnet_downloader.wordnet_path))
        self.assertEqual(wordnet_downloader.url_hungarian_wordnet,
                         "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/HuWN.zip")
        self.assertIn("HuWN.zip", str(wordnet_downloader.wordnet_zip_file))

    def test_check_existing_file(self):
        wordnet_downloader = WordNetDownloader()
        self.assertEqual(wordnet_downloader.download_needed, False)
        wordnet_downloader.check_existing_file()
        download_needed = not os.path.isfile(wordnet_downloader.wordnet_path)
        self.assertEqual(download_needed, wordnet_downloader.download_needed)

    def test_download_cleanup(self):
        wordnet_downloader = WordNetDownloader()
        wordnet_downloader.download()
        self.assertTrue(os.path.isfile(wordnet_downloader.wordnet_zip_file))
        wordnet_downloader.cleanup()
        self.assertFalse(os.path.isfile(wordnet_downloader.wordnet_zip_file))

    def test_run(self):
        wordnet_downloader = WordNetDownloader()
        os.remove(wordnet_downloader.wordnet_path)
        wordnet_downloader.run()
        self.assertFalse(os.path.isfile(wordnet_downloader.wordnet_zip_file))
        self.assertTrue(os.path.isfile(wordnet_downloader.wordnet_path))

if __name__ == '__main__':
    unittest.main()
