"""Imports Datasets"""
import csv
import glob
import os
import urllib
import zipfile
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import academictorrents as at
from data.utils import symbol_map, ensg_to_hugo_map



class GeneDataset(Dataset):
    """Gene Expression Dataset."""
    def __init__(self):
        self.load_data()

    def load_data(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class TCGADataset(GeneDataset):
    def __init__(self, nb_examples=None, at_hash="e4081b995625f9fc599ad860138acf7b6eb1cf6f", datastore=""):
        self.at_hash = at_hash
        self.datastore = datastore
        self.nb_examples = nb_examples # In case you don't want to load the whole dataset from disk
        super(TCGADataset, self).__init__()

    def load_data(self):
        csv_file = at.get(self.at_hash, datastore=self.datastore)
        hdf_file = csv_file.split(".gz")[0] + ".hdf5"
        if not os.path.isfile(hdf_file):
            print("We are converting a CSV dataset of TCGA to HDF5. Please wait a minute, this only happens the first "
                  "time you use the TCGA dataset.")
            df = pd.read_csv(csv_file, compression="gzip", sep="\t")
            df = df.set_index('Sample')
            df = df.transpose()
            df.to_hdf(hdf_file, key="data", complevel=5)
        self.df = pd.read_hdf(hdf_file)
        self.df.rename(symbol_map(self.df.columns), axis="columns", inplace=True)
        self.df = self.df - self.df.mean(axis=0)
        #self.df = self.df / self.df.variance()
        self.sample_names = self.df.index.values.tolist()
        self.node_names = np.array(self.df.columns.values.tolist()).astype("str")
        self.nb_nodes = self.df.shape[1]
        self.labels = [0 for _ in range(self.df.shape[0])]

    def __getitem__(self, idx):
        sample = np.array(self.df.iloc[idx])
        sample = np.expand_dims(sample, axis=-1)
        label = self.labels[idx]
        sample = {'sample': sample, 'labels': label}
        return sample
