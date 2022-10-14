import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from scipy import signal

PATH_TO_DATA = '../Data'
SUBJECTS = list(range(9))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, colab):
        if colab:
            self.X = torch.cuda.FloatTensor(X)
            self.Y = torch.cuda.LongTensor(Y)
        else:
            self.X = torch.FloatTensor(X)
            self.Y = torch.LongTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y


def load_data(data_path=PATH_TO_DATA, subjects=SUBJECTS, sample_mult=4, verbose=True):
    X_test = np.load(f'{data_path}/X_test.npy')
    y_test = np.load(f'{data_path}/y_test.npy') - 769
    person_train_valid = np.load(f'{data_path}/person_train_valid.npy')
    X_train_valid = np.load(f'{data_path}/X_train_valid.npy')
    y_train_valid = np.load(f'{data_path}/y_train_valid.npy') - 769
    person_test = np.load(f'{data_path}/person_test.npy')

    X_train_valid_subjects = np.empty((0, X_train_valid.shape[1], X_train_valid.shape[2]))
    y_train_valid_subjects = np.empty((0))
    X_test_subjects = np.empty((0, X_test.shape[1], X_test.shape[2]))
    y_test_subjects = np.empty((0))

    for s in subjects:
        train_valid_subject_idx = np.where(person_train_valid == s)[0]
        person_test_subject_idx = np.where(person_test == s)[0]
        X_train_valid_subject = X_train_valid[train_valid_subject_idx, :, :]
        y_train_valid_subject = y_train_valid[train_valid_subject_idx]
        X_test_subject = X_test[person_test_subject_idx, :, :]
        y_test_subject = y_test[person_test_subject_idx]

        # stack
        X_train_valid_subjects = np.vstack((X_train_valid_subjects, X_train_valid_subject))
        y_train_valid_subjects = np.hstack((y_train_valid_subjects, y_train_valid_subject))
        X_test_subjects = np.vstack((X_test_subjects, X_test_subject))
        y_test_subjects = np.hstack((y_test_subjects, y_test_subject))

    if verbose:
        print('Training/Valid data shape: {}'.format(X_train_valid_subjects.shape))
        print('Test data shape: {}'.format(X_test_subjects.shape))

    X_train_valid_prep, y_train_valid_prep = data_prep(X_train_valid_subjects, y_train_valid_subjects, sample_mult, sample_mult, True, verbose=verbose)
    X_test_prep, y_test_prep = data_prep(X_test_subjects, y_test_subjects, sample_mult, sample_mult, True, verbose=verbose)

    if verbose:
        print(X_train_valid_prep.shape)
        print(y_train_valid_prep.shape)
        print(X_test_prep.shape)
        print(y_test_prep.shape)

    return X_train_valid_prep, y_train_valid_prep, X_test_prep, y_test_prep


def setup_data(X_train_valid, y_train_valid, X_test, y_test, batch_size, colab=True, verbose=True):
    n_data = X_train_valid.shape[0]
    ind_valid = np.random.choice(n_data, int(n_data*.2), replace=False)
    ind_train = np.array(list(set(range(n_data)).difference(set(ind_valid))))

    if verbose:
        print(X_train_valid.shape)

    (X_train, X_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
    if verbose:
        print('Shape of training set:',X_train.shape)
        print('Shape of validation set:',X_valid.shape)
        print('Shape of training labels:',y_train.shape)
        print('Shape of validation labels:',y_valid.shape)

    # Adding width of the segment to be 1
    X_train = X_train[:, :, np.newaxis, :]
    X_valid = X_valid[:, :, np.newaxis, :]
    X_test = X_test[:, :, np.newaxis, :]
    if verbose:
        print('Shape of training set after adding width info:', X_train.shape)
        print('Shape of validation set after adding width info:', X_valid.shape)
        print('Shape of test set after adding width info:', X_test.shape)

    # load training dataset
    train_dataset = Dataset(X_train, y_train, colab)
    train_loader = DataLoader(train_dataset, batch_size)

    # load validation dataset
    val_dataset = Dataset(X_valid, y_valid, colab)
    val_loader = DataLoader(val_dataset, len(val_dataset))

    # load test dataset
    test_dataset = Dataset(X_test, y_test, colab)
    test_loader = DataLoader(test_dataset, len(test_dataset))

    return [train_loader, val_loader, test_loader]

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y.astype(np.uint8)]

# PREPROCESSING

def data_prep(X,y,sub_sample,average,noise, verbose=True):
    
    total_X = None
    total_y = None
    
    # Trimming the data
    X = X[:,:,0:(X.shape[2]-(X.shape[2]%sub_sample))]
    if verbose:
        print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    if verbose:
        print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    if verbose:
        print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    if verbose:
        print('Shape of X after subsampling and concatenating:',total_X.shape)
    
    return total_X,total_y