from __future__ import print_function, division
import warnings; warnings.filterwarnings("ignore")

from nilmtk import DataSet
import pandas as pd
import numpy as np

import datetime
import time
import math
import glob

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam, Nadam, RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# Bring packages onto the path
import sys, os
sys.path.append(os.path.abspath('../bayesian_optimization/'))

from utils import metrics_np
from utils.metrics_np import Metrics


def array_layers(num_layer):
    """
    function to create array to be input for build_fc_model()
    eg. array_layers(5) --> [1, 256, 512, 1024, 1]
    """
    output_array = []
    current_hidden_nodes = 256
    for i in range(1,num_layer+1):
        # first layer
        if i == 1:
            output_array.append(1)
        elif i==(num_layer):
            output_array.append(1)
        else:
            output_array.append(current_hidden_nodes)
            current_hidden_nodes *= 2
    return output_array

def build_fc_model(layers, dropout_prob=0.5):
    fc_model = Sequential()
    for i in range(len(layers)-1):
        fc_model.add( Dense(input_dim=layers[i], output_dim= layers[i+1]) )#, W_regularizer=l2(0.1)) )
        fc_model.add( Dropout(dropout_prob) )
        if i < (len(layers) - 2):
            fc_model.add( Activation('relu') )
    fc_model.summary()
    return fc_model

def fcnn(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end, test_building, test_start, test_end, meter_key, sample_period, num_epochs,
                    patience,
                    num_layers,
                    optimizer,
                    learning_rate,
                    dropout_prob,
                    loss):

    # Start tracking time
    start = time.time()

    # Prepare dataset and options
    dataset_path = dataset_path
    train = DataSet(dataset_path)
    train.set_window(start=train_start, end=train_end)
    val = DataSet(dataset_path)
    val.set_window(start=val_start, end=val_end)
    test = DataSet(dataset_path)
    test.set_window(start=test_start, end=test_end)
    train_building = train_building
    val_building = val_building
    test_building = test_building
    meter_key = meter_key

    sample_period = sample_period


    train_elec = train.buildings[train_building].elec
    val_elec = val.buildings[val_building].elec
    test_elec = test.buildings[test_building].elec

    try: # REDD
        X_train = next(train_elec.mains().all_meters()[0].load(sample_period=sample_period)).fillna(0)
        y_train = next(train_elec[meter_key].load(sample_period=sample_period)).fillna(0)
        X_test = next(test_elec.mains().all_meters()[0].load(sample_period=sample_period)).fillna(0)
        y_test = next(test_elec[meter_key].load(sample_period=sample_period)).fillna(0)
        X_val = next(val_elec.mains().all_meters()[0].load(sample_period=sample_period)).fillna(0)
        y_val = next(val_elec[meter_key].load(sample_period=sample_period)).fillna(0)

        # Intersect between two dataframe - to make sure same trining instances in X and y
        # Train set
        intersect_index = pd.Index(np.sort(list(set(X_train.index).intersection(set(y_train.index)))))
        X_train = X_train.ix[intersect_index]
        y_train = y_train.ix[intersect_index]
        # Test set
        intersect_index = pd.Index(np.sort(list(set(X_test.index).intersection(set(y_test.index)))))
        X_test = X_test.ix[intersect_index]
        y_test = y_test.ix[intersect_index]
         # Val set
        intersect_index = pd.Index(np.sort(list(set(X_val.index).intersection(set(y_val.index)))))
        X_val = X_val.ix[intersect_index]
        y_val = y_val.ix[intersect_index]

        # Get values from numpy array
        X_train = X_train.values
        y_train = y_train.values
        X_test = X_test.values
        y_test = y_test.values
        X_val = X_val.values
        y_val = y_val.values
    except AttributeError: # UKDALE
        X_train = train_elec.mains().power_series_all_data(sample_period=sample_period).fillna(0)
        y_train = next(train_elec[meter_key].power_series(sample_period=sample_period)).fillna(0)
        X_test = test_elec.mains().power_series_all_data(sample_period=sample_period).fillna(0)
        y_test = next(test_elec[meter_key].power_series(sample_period=sample_period)).fillna(0)

        # Intersect between two dataframe - to make sure same trining instances in X and y
        # Train set
        intersect_index = pd.Index(np.sort(list(set(X_train.index).intersection(set(y_train.index)))))
        X_train = X_train.ix[intersect_index]
        y_train = y_train.ix[intersect_index]
        # Test set
        intersect_index = pd.Index(np.sort(list(set(X_test.index).intersection(set(y_test.index)))))
        X_test = X_test.ix[intersect_index]
        y_test = y_test.ix[intersect_index]

        # X_train = X_train.reshape(-1, 1)
        # y_train = y_train.reshape(-1, 1)
        # X_test = X_test.reshape(-1, 1)
        # y_test = y_test.reshape(-1, 1)

        # Get values from numpy array - Avoid server error
        X_train = X_train.values.reshape(-1, 1)
        y_train = y_train.values.reshape(-1, 1)
        X_test = X_test.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)


    # Model settings and hyperparameters
    layers_array = array_layers(num_layers)
    fc_model = build_fc_model(layers_array, dropout_prob)
    # adam = Adam(lr = 1e-5)
    optimizer = optimizer(lr = learning_rate)
    fc_model.compile(loss=loss, optimizer=optimizer)

    # print("========== TRAIN ============")
    #checkpointer = ModelCheckpoint(filepath="results/fcnn-model-{}-{}epochs.h5".format(meter_key, num_epochs), verbose=0, save_best_only=True)

    # Early stopping when validation loss increases
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=patience,
                              verbose=0, mode='auto')

    hist_fc_ = fc_model.fit( X_train, y_train,
                                batch_size=512, verbose=1, nb_epoch=num_epochs,
                                validation_split=0.2, shuffle=True, callbacks=[earlystop])#  , checkpointer])

    # Get number of earlystop epochs
    num_epochs = earlystop.stopped_epoch if earlystop.stopped_epoch != 0 else num_epochs


    # print("========== DISAGGREGATE ============")
    val_pred_fc = fc_model.predict(X_val).reshape(-1)
    test_pred_fc = fc_model.predict(X_test).reshape(-1)

    # print("========== RESULTS ============")
    # me = Metrics(state_boundaries=[10])
    on_power_threshold = train_elec[meter_key].on_power_threshold()
    me = Metrics(state_boundaries=[on_power_threshold])
    val_metrics_results_dict = Metrics.compute_metrics(me, val_pred_fc, y_val.flatten())
    test_metrics_results_dict = Metrics.compute_metrics(me, test_pred_fc, y_test.flatten())

    # end tracking time
    end = time.time()

    time_taken = end-start # in seconds

    # model_result_data = {
    #     'algorithm_name': 'FCNN',
    #     'datapath': dataset_path,
    #     'train_building': train_building,
    #     'train_start': str(train_start.date()) if train_start != None else None ,
    #     'train_end': str(train_end.date()) if train_end != None else None ,
    #     'test_building': test_building,
    #     'test_start': str(test_start.date()) if test_start != None else None ,
    #     'test_end': str(test_end.date()) if test_end != None else None ,
    #     'appliance': meter_key,
    #     'sampling_rate': sample_period,
    #
    #     'algorithm_info': {
    #         'options': {
    #             'epochs': num_epochs
    #         },
    #         'hyperparameters': {
    #             'sequence_length': None,
    #             'min_sample_split': None,
    #             'num_layers': num_layers
    #         },
    #         'profile': {
    #             'parameters': None
    #         }
    #     },
    #
    #     'metrics':  metrics_results_dict,
    #
    #     'time_taken': format(time_taken, '.2f'),
    # }

    model_result_data = {
        'val_metrics':  val_metrics_results_dict,
        'test_metrics':  test_metrics_results_dict,
        'time_taken': format(time_taken, '.2f'),
        'epochs': num_epochs,
    }

    # Close Dataset files
    train.store.close()
    test.store.close()

    return model_result_data
#
# def main():
#
#     # Take in arguments from command line
#     parser = argparse.ArgumentParser(description='FCNN')
#     parser.add_argument('--datapath', '-d', type=str,  required=True,
#                         help='hd5 filepath')
#
#     parser.add_argument('--train_building', type=int, required=True)
#     parser.add_argument('--train_start', type=str, default=None, help='YYYY-MM-DD')
#     parser.add_argument('--train_end', type=str, required=True, help='YYYY-MM-DD')
#
#     parser.add_argument('--test_building', type=int, required=True)
#     parser.add_argument('--test_start', type=str, required=True, help='YYYY-MM-DD')
#     parser.add_argument('--test_end', type=str, default=None, help='YYYY-MM-DD')
#
#     parser.add_argument('--appliance', type=str, required=True)
#     parser.add_argument('--sampling_rate', type=int, required=True)
#
#     # Model specific options and hyperparameters
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--num_layers', type=int, default=5)
#     args = parser.parse_args()
#
#     hd5_filepath = args.datapath
#     train_building = args.train_building
#     train_start = pd.Timestamp(args.train_start) if args.train_start != None else None
#     train_end = pd.Timestamp(args.train_end)
#     test_building = args.test_building
#     test_start = pd.Timestamp(args.test_start)
#     test_end = pd.Timestamp(args.test_end) if args.test_end != None else None
#     appliance = args.appliance
#     downsampling_period = args.sampling_rate
#     epochs = args.epochs
#     num_layers = args.num_layers
#
#
#     model_result_data = fcnn(
#         dataset_path=hd5_filepath,
#         train_building=train_building, train_start=train_start, train_end=train_end,
#         test_building=test_building, test_start=test_start, test_end=test_end,
#         meter_key=appliance,
#         sample_period=downsampling_period,
#         num_epochs=epochs,
#         num_layers=num_layers)
#
#     # Write options and results to file
#     with open('results/fcnn_json.json', 'a+') as outfile:
#         json.dump(model_result_data, outfile, sort_keys=True,
#                  indent=4, separators=(',', ': '))
#     print(model_result_data)
#
# if __name__ == "__main__":
#     main()
#
#     # python fcnn.py --datapath ../data/REDD/redd.h5 --train_building 1 --train_building 1 --train_end 2011-05-10 --test_building 1 --test_start 2011-05-10 --appliance fridge --sampling_rate 20 --epochs 1 --num_layers 5
#     # python fcnn.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-10 --test_building 1 --test_start 2011-05-10 --appliance fridge --sampling_rate 20 --epochs 1 --num_layers 5
