import warnings; warnings.filterwarnings("ignore")

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

# For surpressing print
import os, sys

print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath('bayesian_optimization/'))
prepend_path = "bayesian_optimization/"

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

from nilmtk import DataSet
import pandas as pd
import numpy as np

from keras.optimizers import Adam, Nadam, RMSprop

import datetime
import time
import math
import glob

from sklearn.tree import DecisionTreeRegressor

import argparse

import json

# Import algorithms
from algorithms.dt import decision_tree
from algorithms.co import combinatorial_optimisation
from algorithms.fhmm import fhmm
from algorithms.randomforest import random_forest
from algorithms.fcnn import fcnn
from algorithms.gru import gru
from algorithms.lstm import lstm
from algorithms.dae import dae

from utils import metrics_np
from utils.metrics_np import Metrics

#######################################################
################## Function for reversing metrics for minimization
#######################################################
def metrics_minmax_reverse(metric_results, metrics_to_optimize):
    metric_to_reverse = ['precision_score', 'accuracy_score', 'f1_score', 'disaggregation_accuracy']
    if metrics_to_optimize in metric_to_reverse:
        # return negative to maximize a metric instead
        return -1*metric_results[metrics_to_optimize]
    else:
        return metric_results[metrics_to_optimize]

def metrics_minmax_reverse_print(metric, metrics_to_optimize):
    metric_to_reverse = ['precision_score', 'accuracy_score', 'f1_score', 'disaggregation_accuracy']
    if metrics_to_optimize in metric_to_reverse:
        # return negative to maximize a metric instead
        return -1*metric
    else:
        return metric

#######################################################
################## Global Variables
#######################################################
count = 0
best = 0

hd5_filepath = None
train_building = None
train_start = None
train_end = None
test_building = None
test_start = None
test_end = None
appliance = None
downsampling_period = None

metrics_to_optimize = None
num_epochs = None
patience = None

#######################################################
################## Main objective function for hyperopt
#######################################################
def objective(args):
    global best, count, num_epochs, patience, metrics_to_optimize
    count += 1

    algorithm = args['type']

    with HiddenPrints():
    # if True:

        try:
            if algorithm == 'decision tree':

                criterion = args['criterion']
                min_sample_split = int(args['min_sample_split'])

                model_result_data = decision_tree(
                    dataset_path=hd5_filepath,
                    train_building=train_building, train_start=train_start, train_end=train_end,
                    test_building=test_building, test_start=test_start, test_end=test_end,
                    meter_key=appliance,
                    sample_period=downsampling_period,
                    criterion=criterion,
                    min_sample_split=min_sample_split)

            elif algorithm == 'random forest':

                n_estimators = int(args['n_estimators'])
                criterion = args['criterion']
                min_samples_split = int(args['min_samples_split'])

                model_result_data = random_forest(
                    dataset_path=hd5_filepath,
                    train_building=train_building, train_start=train_start, train_end=train_end,
                    test_building=test_building, test_start=test_start, test_end=test_end,
                    meter_key=appliance,
                    sample_period=downsampling_period,
                    n_estimators=n_estimators,
                    criterion=criterion,
                    min_sample_split=min_samples_split)

            elif algorithm == 'combinatorial optimization':
                model_result_data = combinatorial_optimisation(
                            dataset_path=hd5_filepath,
                            train_building=train_building, train_start=train_start, train_end=train_end,
                            test_building=test_building, test_start=test_start, test_end=test_end,
                            meter_key=appliance,
                            sample_period=downsampling_period)

            elif algorithm == 'factorial hidden markov models':
                model_result_data = fhmm(
                        dataset_path=hd5_filepath,
                        train_building=train_building, train_start=train_start, train_end=train_end,
                        test_building=test_building, test_start=test_start, test_end=test_end,
                        meter_key=appliance,
                        sample_period=downsampling_period)

            elif algorithm == 'fully-connected neural networks':

                num_layers = int(args['num_layers'])
                optimizer = args['optimizer']
                learning_rate = args['learning_rate']
                dropout_prob = args['dropout_prob']
                loss_function = args['loss_function']

                model_result_data = fcnn(
                        dataset_path=hd5_filepath,
                        train_building=train_building, train_start=train_start, train_end=train_end,
                        test_building=test_building, test_start=test_start, test_end=test_end,
                        meter_key=appliance,
                        sample_period=downsampling_period,
                        num_epochs=num_epochs, # TODO: make it params?
                        patience=patience,
                        num_layers=num_layers,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        dropout_prob=dropout_prob,
                        loss=loss_function)

            elif algorithm == 'gated recurrent units':

                optimizer = args['optimizer']
                learning_rate = args['learning_rate']
                loss_function = args['loss_function']

                model_result_data = gru(
                        dataset_path=hd5_filepath,
                        train_building=train_building, train_start=train_start, train_end=train_end,
                        test_building=test_building, test_start=test_start, test_end=test_end,
                        meter_key=appliance,
                        sample_period=downsampling_period,
                        num_epochs=num_epochs, # TODO: make it params?
                        patience=patience,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        loss=loss_function)

            elif algorithm == 'long short-term memory':

                optimizer = args['optimizer']
                learning_rate = args['learning_rate']
                loss_function = args['loss_function']

                model_result_data = lstm(
                        dataset_path=hd5_filepath,
                        train_building=train_building, train_start=train_start, train_end=train_end,
                        test_building=test_building, test_start=test_start, test_end=test_end,
                        meter_key=appliance,
                        sample_period=downsampling_period,
                        num_epochs=num_epochs, # TODO: make it params?
                        patience=patience,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        loss=loss_function)

            elif algorithm == 'denoising autoencoders':
                sequence_length = args['sequence_length']
                optimizer = args['optimizer']
                learning_rate = args['learning_rate']
                loss_function = args['loss_function']

                model_result_data = dae(
                        dataset_path=hd5_filepath,
                        train_building=train_building, train_start=train_start, train_end=train_end,
                        test_building=test_building, test_start=test_start, test_end=test_end,
                        meter_key=appliance,
                        sample_period=downsampling_period,
                        num_epochs=num_epochs, # TODO: make it params?
                        patience=patience,
                        sequence_length=sequence_length,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        loss=loss_function)
        except:
             # Convert keras optimizer type to String
            if 'optimizer' in args:
                args['optimizer'] = args['optimizer'].__name__

            results = {
                        'args': args,
                        'status': STATUS_FAIL}
            with open(prepend_path+'results/trials_temp.json', 'a') as f:
                json.dump(results, f)
                f.write(os.linesep)
            return results


     ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    # Convert keras optimizer type to String
    if 'optimizer' in args:
        args['optimizer'] = args['optimizer'].__name__

    # Extract info from model_result_data
    metrics = model_result_data['metrics']
    time_taken = model_result_data['time_taken']
    epochs = model_result_data['epochs']


    # Print progress - Reguarly
    print ('iters:', count, ', ',metrics_to_optimize,':', metrics[metrics_to_optimize], 'using', args['type'])

    if count == 1:
        print('new best:', metrics[metrics_to_optimize], 'using', args['type'])
        best = metrics_minmax_reverse(metrics, metrics_to_optimize)
    elif metrics_minmax_reverse(metrics, metrics_to_optimize) < best:
        print ('new best:', metrics[metrics_to_optimize], 'using', args['type'])
        best = metrics_minmax_reverse(metrics, metrics_to_optimize)

    # Write trial result to file
    results = {
            'args': args,
            'loss': metrics[metrics_to_optimize], # return normall loss without need to inverse for maximizing
            'metrics': metrics,
            'time_taken': time_taken,
            'epochs': epochs,
            'status': STATUS_OK,
            'order': count,
            }
    with open(prepend_path+'results/trials_temp.json', 'a') as f:
        json.dump(results, f)
        f.write(os.linesep)

    return {
            'args': args,
            'loss': metrics_minmax_reverse(metrics, metrics_to_optimize), # Need to inverse for maximizing for fmin()
            'metrics': metrics,
            'time_taken': time_taken,
            'epochs': epochs,
            'status': STATUS_OK,
            'order': count,
            }
#######################################################
################## Main function
#######################################################
def main(args_hd5_filepath ,args_train_building, args_train_start, args_train_end, args_test_building , args_test_start ,args_test_end ,args_appliance ,args_downsampling_period, args_max_evals, args_metrics_to_optimize, args_num_epochs, args_patience):

    global hd5_filepath ,train_building, train_start, train_end, test_building , test_start ,test_end ,appliance ,downsampling_period
    global metrics_to_optimize, num_epochs, patience

    hd5_filepath = args_hd5_filepath
    train_building = args_train_building
    train_start = pd.Timestamp(args_train_start) if args_train_start != None else None
    train_end = pd.Timestamp(args_train_end)
    test_building = args_test_building
    test_start = pd.Timestamp(args_test_start)
    test_end = pd.Timestamp(args_test_end) if args_test_end != None else None
    appliance = args_appliance
    downsampling_period = args_downsampling_period
    # epochs = args.epochs
    num_epochs = args_num_epochs
    patience = args_patience
    max_evals = args_max_evals
    metrics_to_optimize = args_metrics_to_optimize

    # Search space
    space = hp.choice('algorithm', [
        {
            'type': 'decision tree',
            'criterion': hp.choice('dtree_criterion', ["mse", "friedman_mse", "mae"]),
            'min_sample_split': hp.quniform('dtree_min_samples_split', 2, 200, 1),
        },
        {
            'type': 'random forest',
            'n_estimators': hp.quniform('randforest_n_estimators', 5, 100, 1),
            'criterion': hp.choice('randforest_criterion', ["mse", "friedman_mse", "mae"]),
            'min_samples_split': hp.quniform('randforest_min_samples_split', 2, 200, 1),
        },
        {
            'type': 'fully-connected neural networks',
    #         'num_layers': hp.quniform('fcnn_num_layers', 5, 8, 1),
            'num_layers': hp.quniform('fcnn_num_layers', 5, 7, 1), #TODO: change back to 5,8 only for testing
            'optimizer': hp.choice('fcnn_optimizer', [Adam ,Nadam, RMSprop]),
            'learning_rate': hp.choice('fcnn_learning_rate', [0.00001, 0.0001, 0.001, 0.01]),
            'dropout_prob': hp.uniform('fcnn_dropout_prob', 0.1, 0.6),
            'loss_function': hp.choice('fcnn_loss_function', ["mean_squared_error", "mean_absolute_error"]),
        },
        {
            'type': 'combinatorial optimization',
        },
        {
            'type': 'factorial hidden markov models',
        },
        {
            'type': 'gated recurrent units',
            'optimizer': hp.choice('gru_optimizer', [Adam ,Nadam, RMSprop]),
            'learning_rate': hp.choice('gru_learning_rate', [0.00001, 0.0001, 0.001, 0.01]),
            'loss_function': hp.choice('gru_loss_function', ["mean_squared_error", "mean_absolute_error"]),
        },
        {
            'type': 'long short-term memory',
            'optimizer': hp.choice('lstm_optimizer', [Adam ,Nadam, RMSprop]),
            'learning_rate': hp.choice('lstm_learning_rate', [0.00001, 0.0001, 0.001, 0.01]),
            'loss_function': hp.choice('lstm_loss_function', ["mean_squared_error", "mean_absolute_error"]),
        },
        {
            'type': 'denoising autoencoders',
            'sequence_length': hp.choice('dae_sequence_length', [64, 128, 256, 512, 1024]),
            'optimizer': hp.choice('dae_optimizer', [Adam ,Nadam, RMSprop]),
            'learning_rate': hp.choice('dae_learning_rate', [0.00001, 0.0001, 0.001, 0.01]),
            'loss_function': hp.choice('dae_loss_function', ["mean_squared_error", "mean_absolute_error"]),
        },
    ])

    #######################################################
    ###### Start to Optimize
    #######################################################
    # Delete temp file if already exist
    try:
        os.remove(prepend_path+'results/trials_temp.json')
    except OSError:
        pass

    # Start tracking time
    trial_start_date = str(datetime.datetime.now().date())
    trial_start_time = str(datetime.datetime.now().time())
    start = time.time()

    # Minimize the objective over the space
    trials = Trials()
    try:
        best_params = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, verbose=1, trials=trials)

        print("*"*40)
        # print("Best evalution is: ", space_eval(space, best_params), " with ", metrics_to_optimize, " of: ", best)
        print("Best evalution is: ", space_eval(space, best_params), " with ", metrics_to_optimize, " of: ", metrics_minmax_reverse_print(best, metrics_to_optimize))
    except Exception as e:
        print("Error with fmin() function!!!")
        print(e)

    # end tracking time
    end = time.time()
    time_taken = end-start # in seconds

    #######################################################
    ################## Write Trials Results to File
    #######################################################
    # Write metadata for Hyperopt to file

    space = space_eval(space, best_params)
    # Convert keras optimizer type to String
    if 'optimizer' in space:
        space['optimizer'] = space['optimizer'].__name__

    trial_metadata = {
        'hd5_filepath': hd5_filepath,
        'train_building': train_building,
        'train_start': str(train_start.date()) if train_start != None else None ,
        'train_end': str(train_end.date()) if train_end != None else None ,
        'test_building': test_building,
        'test_start': str(test_start.date()) if test_start != None else None ,
        'test_end': str(test_end.date()) if test_end != None else None ,
        'appliance': appliance,
        'downsampling_period': downsampling_period,
        'num_epochs': num_epochs,
        'patience': patience,
        'max_evals': max_evals,
        'metrics_to_optimize': metrics_to_optimize,
        'trial_start_date': trial_start_date,
        'trial_start_time': trial_start_time,
        'time_taken': format(time_taken, '.2f'),
        'space': space,
        'loss': metrics_minmax_reverse_print(best, metrics_to_optimize)
    }
    with open(prepend_path+'results/trial-metadata-{}-{}-{}.json'.format(appliance, downsampling_period,metrics_to_optimize), 'w+') as outfile:
        json.dump(trial_metadata, outfile,
                 indent=4, separators=(',', ': '))

    # Open trials temp file
    with open(prepend_path+'results/trials_temp.json') as f:
        trial_list = [json.loads(line) for line in f]

    # Write options and results to file
    with open(prepend_path+'results/trial-results-{}-{}-{}.json'.format(appliance, downsampling_period,metrics_to_optimize), 'w+') as outfile:
        json.dump(trial_list, outfile,
                 indent=4, separators=(',', ': '))

    # Files for displaying in FLASK
    # Delete existing files if exist
    try:
        os.remove(prepend_path+'results/latest_metadata.json')
        os.remove(prepend_path+'results/latest_results.json')
    except OSError:
        pass
    # Write metadata and Results of most recent trails
    with open(prepend_path+'results/latest_metadata.json'.format(appliance, downsampling_period,metrics_to_optimize), 'w+') as outfile:
        json.dump(trial_metadata, outfile,
                 indent=4, separators=(',', ': '))

    with open(prepend_path+'results/latest_results.json'.format(appliance, downsampling_period,metrics_to_optimize), 'w+') as outfile:
        json.dump(trial_list, outfile,
                 indent=4, separators=(',', ': '))
