from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import time

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from algorithms.GRU.grudisaggregator import GRUDisaggregator

import pandas as pd

# Bring packages onto the path
import sys, os
sys.path.append(os.path.abspath('../bayesian_optimization/'))

from utils import metrics


def gru(dataset_path, train_building, train_start, train_end, val_building, val_start, val_end, test_building, test_start, test_end, meter_key, sample_period, num_epochs, patience, optimizer, learning_rate, loss):

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

    train_meter = train_elec.submeters()[meter_key]
    try:
        train_mains = train_elec.mains().all_meters()[0]
        val_mains = val_elec.mains().all_meters()[0]
        test_mains = test_elec.mains().all_meters()[0]
    except AttributeError:
        train_mains = train_elec.mains()
        test_mains = test_elec.mains()


    gru = GRUDisaggregator(patience, optimizer, learning_rate, loss)
    print(gru.model.summary())

    # print("========== TRAIN ============")
    gru.train(train_mains, train_meter, epochs=num_epochs, sample_period=sample_period)

    # Get number of earlystop epochs
    num_epochs = gru.stopped_epoch if gru.stopped_epoch != 0 else num_epochs

    #gru.export_model("results/gru-model-{}-{}epochs.h5".format(meter_key, num_epochs))


    # print("========== DISAGGREGATE ============")
    # Validation
    val_disag_filename = 'disag-out-val.h5'
    output = HDFDataStore(val_disag_filename, 'w')
    gru.disaggregate(val_mains, output, train_meter, sample_period=sample_period)
    output.close()
    # Test
    test_disag_filename = 'disag-out-test.h5'
    output = HDFDataStore(test_disag_filename, 'w')
    gru.disaggregate(test_mains, output, train_meter, sample_period=sample_period)
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
    #     'algorithm_name': 'GRU',
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
        'epochs': num_epochs,
    }

    # Close digag_filename
    result.store.close()
    result_val.store.close()

    # Close Dataset files
    train.store.close()
    val.store.close()
    test.store.close()

    return model_result_data

# def main():
#
#     # Take in arguments from command line
#     parser = argparse.ArgumentParser(description='GRU')
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
#
#
#     model_result_data = gru(
#         dataset_path=hd5_filepath,
#         train_building=train_building, train_start=train_start, train_end=train_end,
#         test_building=test_building, test_start=test_start, test_end=test_end,
#         meter_key=appliance,
#         sample_period=downsampling_period,
#         num_epochs=epochs)
#
#     # Write options and results to file
#     with open('results/gru_json.json', 'a+') as outfile:
#         json.dump(model_result_data, outfile, sort_keys=True,
#                  indent=4, separators=(',', ': '))
#     print(model_result_data)
#
# if __name__ == "__main__":
#     main()

    # python gru.py --datapath ../data/REDD/redd.h5 --train_building 1 --train_building 1 --train_end 2011-05-10 --test_building 1 --test_start 2011-05-10 --appliance fridge --sampling_rate 20 --epochs 1
    # python gru.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-10 --test_building 1 --test_start 2011-05-10 --appliance fridge --sampling_rate 20 --epochs 1
