import numpy as np
import pandas as pd
import os
import sys
import gc
from datetime import datetime
from tqdm import tqdm_notebook, tnrange
from scipy import signal
import matplotlib.pyplot as plt
from mltools.utils import load_hparams, get_logger
from mltools.models.gbm_model import GBMModel

path_dict = {'win32': r'c:\lanl_data',
             'darwin': '/users/user/lanl_data',
             'linux': '~/lanl_data'}
data_path = path_dict[sys.platform]

def load_fold_data(filename):
    import pickle
    with open(os.path.join(data_path, filename), 'rb') as file:
        [X_train, y_train, X_test] = pickle.load(file)
        return X_train, y_train, X_test


def save_predictions(quake, seg_ids, subs_filename):
    subs = quake.test_predictions()
    subs = pd.Series(subs['lanl_xgb_pred']).rename('time_to_failure')
    subs = pd.concat([seg_ids, subs], axis=1)
    subs.to_csv(subs_filename, index=False)
    quake.logger.info(f'subs shape: {subs.shape}.  saved as {subs_filename}')


def get_weights(y_train, low_cut, high_cut, low_rate, high_rate):
    high_weights = np.where(y_train>high_cut, (y_train-high_cut) * high_rate, 0)
    low_weights = np.where(y_train<low_cut, (low_cut - y_train) * low_rate, 0)
    return np.ones(len(y_train)) + low_weights + high_weights


def main():
    library, params_file = 'xgb', 'quake_xgbtree.yaml'

    logger = get_logger('quake_' + library)
    filename = 'quake_feats.pkl'
    logger.info(f'using {filename}')
    X_train, y_train, X_test = load_fold_data(filename)

#    weights_params = (2, 8, 2/2, 2/8)
#    weights = get_weights(y_train, *weights_params)
#    logger.info(f'weights: {weights_params}')
    weights = None

    seg_ids = X_test.pop('seg_id')
    folds = X_train.pop('fold')

    quake = GBMModel(X_train, y_train, X_test, params_file,
        folds, 'lanl', weights, library, logger)

    subs_filename = 'quake_submissions.csv'
    save_predictions(quake, seg_ids, subs_filename)

if __name__ == '__main__':
    main()

#
