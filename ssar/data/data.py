import datetime
import numpy as np
import os
import pickle

import torch
from torch.utils.data import Sampler


def split_data(number_of_samples, shuffle=True, train_fraction=1.0, validation_fraction=0):
    """
    A function to split data into training, validation, testing partitions. Returns 3 lists of indices each
    corresponding to training_indices, validation_indices, testing_indices respectively

    :parameter
    number_of_samples : int
                Total number of samples in dataset
    shuffle : bool
                If the data should be shuffled or not, defaults to True
    training_fraction : float
                Fraction that should be allocated for training
    validation_fraction : float
                Fraction of data that should allocated for validation
    testing_fraction : float
                Fraction of data that should be allocated to testing

    training_fraction + validation_fraction + testing_fraction should be equal to 1.0, else the behaviour is undefined.

    :returns
    3 lists
        training_indices, validation_indices, testing_indices

    """
    indices = list(range(number_of_samples))

    if shuffle:
        random_seed = 10
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    test_fraction = 1 - (train_fraction + validation_fraction)

    validation_split = int(np.floor(validation_fraction * number_of_samples))
    test_split = int(np.floor(test_fraction * number_of_samples))

    validation_indices = indices[:validation_split]
    test_indices = indices[validation_split:(validation_split+test_split)]
    train_indices = indices[validation_split+test_split:]
    return train_indices, validation_indices, test_indices


def check_and_split_data(host_name, data_folder, dataset_len, train_fraction, validation_fraction):
    """Function to check the metadata folder for existing split indices"""

    meta_data_folder = data_folder + '/.meta_data/'
    meta_data_file = meta_data_folder + "/{}_{}_indices_meta_data.pkl".format("EgoGestData", host_name)
    if os.path.exists(meta_data_folder):
        if os.path.exists(meta_data_file):
            with open(meta_data_file, 'rb') as f:
                print('{}: Opening indices file list from saved metadata folder'.format(datetime.datetime.now().time()))
                meta_data = pickle.load(f)
                training_indices = meta_data['training_indices']
                validation_indices = meta_data['validation_indices']
                testing_indices = meta_data['testing_indices']
                print('{}: Retrieved indices file list from saved metadata folder'.format(datetime.datetime.now().time()))
                return training_indices, validation_indices, testing_indices
    else:
        os.mkdir(meta_data_folder)

    print('{}: Creating indices list '.format(datetime.datetime.now().time()))
    training_indices, validation_indices, testing_indices = split_data(dataset_len, True,
                                                                       train_fraction,
                                                                       validation_fraction)
    with open(meta_data_file, 'wb') as f:
        meta_data = {'training_indices': training_indices,
                     'validation_indices': validation_indices,
                     'testing_indices': testing_indices}
        pickle.dump(meta_data, f)
        print('{}: Saved {} indices list to metadata folder'.format(datetime.datetime.now().time(), meta_data_file))
    return training_indices, validation_indices, testing_indices

class FixedIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __iter__(self):
        return iter(self.indices)

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    sample_batch = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            ## get sequence lengths
            lengths = [sample[key].shape[0] for sample in batch]
            ## padd
            elem_batch = [sample[key] for sample in batch]
            elem_batch = torch.nn.utils.rnn.pad_sequence(elem_batch, batch_first=True)
            elem_batch = torch.nn.utils.rnn.pack_padded_sequence(elem_batch, lengths=lengths, batch_first=True, enforce_sorted=False)
        elif isinstance(batch[0][key], int):
            elem_batch = [sample[key] for sample in batch]
            elem_batch = torch.LongTensor(elem_batch)
        else:
            elem_batch = [sample[key] for sample in batch]
        sample_batch[key] = elem_batch
    return sample_batch