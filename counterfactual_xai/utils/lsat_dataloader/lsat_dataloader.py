import os.path

import requests
import csv


class LsatDataloader:
    LSAT_TEST_FILE = "law_school_cf_test.csv"
    LSAT_TRAIN_FILE = "law_school_cf_train.csv"
    DOWNLOAD_ADRESS = "https://raw.githubusercontent.com/throwaway20190523/MonotonicFairness/master/data/"

    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path

    def _download_lsat_from_github(self, file: str):
        downloadfile = self.LSAT_TEST_FILE if file == "test" else self.LSAT_TRAIN_FILE
        with requests.Session() as s:
            download = s.get(self.DOWNLOAD_ADRESS + downloadfile)
            decoded_content = download.content.decode('utf-8')
            csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')

            with open(str(self.csv_path + downloadfile), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in csv_reader:
                    writer.writerow(row)

    def _check_for_local_lsat_files(self):
        if not os.path.isfile(self.csv_path + self.LSAT_TEST_FILE):
            self._download_lsat_from_github(file="test")

        if not os.path.isfile(self.csv_path + self.LSAT_TRAIN_FILE):
            self._download_lsat_from_github(file="train")

    def get_lsat_dataset(self):
        self._check_for_local_lsat_files()
