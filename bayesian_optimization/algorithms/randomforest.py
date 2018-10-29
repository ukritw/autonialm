from __future__ import print_function, division
import warnings; warnings.filterwarnings("ignore")

from nilmtk import DataSet
import pandas as pd
import numpy as np

import datetime
import time
import math
import glob

from sklearn.ensemble import RandomForestRegressor

# Bring packages onto the path
import sys, os
sys.path.append(os.path.abspath('../bayesian_optimization/'))

from utils import metrics_np
from utils.metrics_np import Metrics


def random_forest(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end, test_building, test_start, test_end, meter_key, sample_period, n_estimators, criterion, min_sample_split):

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
    min_samples_split = min_sample_split
    rf_regr = RandomForestRegressor(n_estimators = n_estimators, criterion = criterion, min_samples_split = min_samples_split, random_state=0)

    # print("========== TRAIN ============")
    rf_regr.fit(X_train, y_train)

    # print("========== DISAGGREGATE ============")
    y_val_predict = rf_regr.predict(X_val)
    y_test_predict = rf_regr.predict(X_test)

    # print("========== RESULTS ============")
    # me = Metrics(state_boundaries=[10])
    on_power_threshold = train_elec[meter_key].on_power_threshold()
    me = Metrics(state_boundaries=[on_power_threshold])
    val_metrics_results_dict = Metrics.compute_metrics(me, y_val_predict, y_val.flatten())
    test_metrics_results_dict = Metrics.compute_metrics(me, y_test_predict, y_test.flatten())

    # end tracking time
    end = time.time()

    time_taken = end-start # in seconds

    # model_result_data = {
    #     'algorithm_name': 'Random Forest Regressor',
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
    #             'epochs': None
    #         },
    #         'hyperparameters': {
    #             'sequence_length': None,
    #             'min_sample_split': min_sample_split,
    #             'num_layers': None
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
        'epochs': None,
    }

    # Close Dataset files
    train.store.close()
    val.store.close()
    test.store.close()

    return model_result_data
