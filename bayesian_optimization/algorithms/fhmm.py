from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import time

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import FHMM

import pandas as pd

# Bring packages onto the path
import sys, os
sys.path.append(os.path.abspath('../bayesian_optimization/'))

from utils import metrics


def fhmm(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end, test_building, test_start, test_end, meter_key, sample_period):

    # Start tracking time
    start = time.time()

    # Prepare dataset and options
    # print("========== OPEN DATASETS ============")
    dataset_path = dataset_path
    train = DataSet(dataset_path)
    train.set_window(start=train_start, end=train_end)
    val = DataSet(dataset_path)
    val.set_window(start=val_start, end=val_end)
    test = DataSet(dataset_path)
    test.set_window(start=test_start, end=test_end)
    train_building = train_building
    test_building = test_building
    meter_key = meter_key

    sample_period = sample_period


    train_elec = train.buildings[train_building].elec
    val_elec = val.buildings[val_building].elec
    test_elec = test.buildings[test_building].elec

    appliances = [meter_key]
    selected_meters = [train_elec[app] for app in appliances]
    selected_meters.append(train_elec.mains())
    selected = MeterGroup(selected_meters)

    fhmm = FHMM()

    # print("========== TRAIN ============")
    fhmm.train(selected, sample_period=sample_period)

    # print("========== DISAGGREGATE ============")
    # Validation
    val_disag_filename = 'disag-out-val.h5'
    output = HDFDataStore(val_disag_filename, 'w')
    fhmm.disaggregate(val_elec.mains(), output_datastore=output)
    output.close()
    # Test
    test_disag_filename = 'disag-out-test.h5'
    output = HDFDataStore(test_disag_filename, 'w')
    fhmm.disaggregate(test_elec.mains(), output_datastore=output)
    output.close()

    # print("========== RESULTS ============")
    # Validation
    result_val = DataSet(val_disag_filename)
    res_elec_val = result_val.buildings[val_building].elec
    rpaf_val = metrics.recall_precision_accuracy_f1(res_elec_val[meter_key], val_elec[meter_key])

    val_metrics_results_dict = {
        'recall_score': rpaf_val[0],
        'precision_score': rpaf_val[1],
        'accuracy_score': rpaf_val[2],
        'f1_score': rpaf_val[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec_val[meter_key], val_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec_val[meter_key], val_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec_val[meter_key], val_elec[meter_key]),
        'nad': metrics.nad(res_elec_val[meter_key], val_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec_val[meter_key], val_elec[meter_key])
        }
    # Test
    result = DataSet(test_disag_filename)
    res_elec = result.buildings[test_building].elec
    rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])

    test_metrics_results_dict = {
        'recall_score': rpaf[0],
        'precision_score': rpaf[1],
        'accuracy_score': rpaf[2],
        'f1_score': rpaf[3],
        'mean_absolute_error': metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key]),
        'mean_squared_error': metrics.mean_square_error(res_elec[meter_key], test_elec[meter_key]),
        'relative_error_in_total_energy': metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key]),
        'nad': metrics.nad(res_elec[meter_key], test_elec[meter_key]),
        'disaggregation_accuracy': metrics.disaggregation_accuracy(res_elec[meter_key], test_elec[meter_key])
        }

    # end tracking time
    end = time.time()

    time_taken = end-start # in seconds

    # model_result_data = {
    #     'algorithm_name': 'FHMM',
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
    #             'min_sample_split': None,
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

    # Close digag_filename
    result.store.close()
    result_val.store.close()

    # Close Dataset files
    train.store.close()
    val.store.close()
    test.store.close()

    return model_result_data
