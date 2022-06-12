import os
from turtle import down
import torch
from glob import glob
import tarfile
import torchaudio
from urllib import request

class Dataset(torch.utils.data.Dataset):
    """creating YES and NO Dataset
    """
    URL = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz'

    def __init__(self, root, tfms=None, download=True):
        super().__init__()

        self.tfms = tfms
        self.path = os.path.join(root, 'yesno.tar.gz')
        self.ext_dest = os.path.join(root, 'waves_yesno')
        if download:
            self.download_dataset(self.URL, self.path)
        self.extract_data(self.path)
        self.file_list = glob(f'{self.ext_dest}/*.wav')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio, labels = self.get_training_pairs(idx)
        audio, sr = torchaudio.load(audio)
        if self.tfms is not None:
            audio = self.tfms(audio)
        audio = audio - audio.min()
        audio = audio / audio.max()
        labels = torch.tensor(labels)
        return audio, labels

    def get_training_pairs(self, idx):
        audio = self.file_list[idx]
        labels = audio.split('/')[-1].split('.')[0].split('_') # extracting labels from filename
        labels = list(map(int, labels))
        return audio, labels

    @staticmethod
    def download_dataset(url, fname):
        print('dataset downloading..')
        request.urlretrieve(url, fname)
        print('dataset downloaded')

    @staticmethod
    def extract_data(fname):
        print('extracting data...')
        with tarfile.open(fname) as f:
            f.extractall()
        print('data extracted')
