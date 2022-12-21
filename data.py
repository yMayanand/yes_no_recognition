import os
from turtle import down
import torch
from glob import glob
import tarfile
import torchaudio
from urllib import request
import math

class Dataset(torch.utils.data.Dataset):
    """creating YES and NO Dataset
    """
    URL = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz'

    def __init__(self, root, train_tfms=None, val_tfms=None,
                 download=True, split=0.33, train=True):
        super().__init__()

        self.train_tfms = train_tfms
        self.val_tfms = val_tfms
        self.split = split
        self.train = train
        self.path = os.path.join(root, 'yesno.tar.gz')
        self.ext_dest = os.path.join(root, 'waves_yesno')
        if download:
            self.download_dataset(self.URL, self.path)
        self.extract_data(self.path)
        self.file_list = glob(f'{self.ext_dest}/*.wav')
        self.train_len = math.floor(self.split*len(self.file_list))
        self.val_len = len(self.file_list) - self.train_len

    def __len__(self):
        return self.train_len if self.train else self.val_len

    def __getitem__(self, idx):
        tfms = self.train_tfms if self.train else self.val_tfms
        idx = idx if self.train else (idx + self.train_len - 1)
        audio, labels = self.get_training_pairs(idx)
        audio, sr = torchaudio.load(audio)
        if tfms is not None:
            audio = tfms(audio)
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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f)
        print('data extracted')
