import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch

def build_dataset(pth, batch_size, shuffle=True):
    #Fetch data from csv file pth
    data = pd.read_csv(pth)

    #Seperate data from target
    X = data.values[:, :-1]
    y = data.values[:, -1]

    #Divide data into batches
    X, y = to_batches(X, y, batch_size, shuffle)

    #Split data into train and validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_batches(X, y, batch_size, shuffle=True):

    if shuffle:
        # Gets the lenght of the table, shuffles it and return a list of the shuffled indexes
        indexes = np.random.permutation(len(x))

        # Arranges the X and y according to the new indices order
        X = X[indexes]
        y = y[indexes]
        
        # Number of batches to be created - rounding to the closest integer
        n_batches = len(X) // batch_size

        # Creates X and y according to the new shape - using np.reshape
        X = X[:n_batches * batch_size].reshape(n_batches, batch_size, X.shape[1])
        y = y[:n_batches * batch_size].reshape(n_batches, batch_size, 1)

    return X, y