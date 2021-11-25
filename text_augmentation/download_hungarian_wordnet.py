# Downloading Hungarian Wordnet from original source
import requests
import os
from importlib_resources import files


class WordNetDownloader:
    def __init__(self):
        self.wordnet_zip_file = files("resources") / "HuWN.zip"
        self.url_hungarian_wordnet = "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/HuWN.zip"
        self.wordnet_path = files("resources") / "HuWN_final4.xml"
        self.download_needed = False

    def check_existing_file(self):
        if not os.path.isfile(self.wordnet_path):
            print("Hungarian Wordnet missing, downloading from original source.")
            self.download_needed = True
        else:
            self.download_needed = False

    def download(self):
        # downloading from original source
        data = requests.get(self.url_hungarian_wordnet, allow_redirects=True)
        with open(self.wordnet_zip_file, 'wb') as file:
            file.write(data.content)

    def unzip(self):
        # unzipping
        os.system('unzip {} -d {}'.format(self.wordnet_zip_file, files("resources")))

    def cleanup(self):
        # deleting zip file
        os.remove(self.wordnet_zip_file)

    def run(self):
        self.check_existing_file()
        if self.download_needed:
            self.download()
            self.unzip()
            self.cleanup()
