from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")

import time

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from algorithms.DAE.daedisaggregator import DAEDisaggregator

import pandas as pd

# Bring packages onto the path
import sys, os
sys.path.append(os.path.abspath('../bayesian_optimization/'))

from utils import metrics


def dae(dataset_path, train_building, train_start, train_end, test_building, test_start, test_end, meter_key, sample_period, num_epochs, patience, sequence_length, optimizer, learning_rate, loss):

    # Start tracking time
    start = time.time()

    # Prepare dataset and options
    # print("========== OPEN DATASETS ============")
    dataset_path = dataset_path
    train = DataSet(dataset_path)
    train.set_window(start=train_start, end=train_end)
    test = DataSet(dataset_path)
    test.set_window(start=test_start, end=test_end)
    train_building = train_building
    test_building = test_building
    meter_key = meter_key

    sample_period = sample_period


    train_elec = train.buildings[train_building].elec
    test_elec = test.buildings[test_building].elec

    train_meter = train_elec.submeters()[meter_key]
    try:
        train_mains = train_elec.mains().all_meters()[0]
        test_mains = test_elec.mains().all_meters()[0]
    except AttributeError:
        train_mains = train_elec.mains()
        test_mains = test_elec.mains()


    dae = DAEDisaggregator(sequence_length, patience, optimizer, learning_rate, loss)

    # print("========== TRAIN ============")
    dae.train(train_mains, train_meter, epochs=num_epochs, sample_period=sample_period)

    # Get number of earlystop epochs
    num_epochs = dae.stopped_epoch if dae.stopped_epoch != 0 else num_epochs

    #dae.export_model("results/dae-model-{}-{}epochs.h5".format(meter_key, num_epochs))


    # print("========== DISAGGREGATE ============")
    disag_filename = 'disag-out.h5'
    output = HDFDataStore(disag_filename, 'w')
    dae.disaggregate(test_mains, output, train_meter, sample_period=sample_period)
    output.close()

    # print("========== RESULTS ============")
    result = DataSet(disag_filename)
    res_elec = result.buildings[test_building].elec
    rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])

    metrics_results_dict = {
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
    #     'algorithm_name': 'DAE',
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
    #             'sequence_length': sequence_length,
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
        'metrics':  metrics_results_dict,
        'time_taken': format(time_taken, '.2f'),
        'epochs': num_epochs,
    }

    # Close digag_filename
    result.store.close()

    # Close Dataset files
    train.store.close()
    test.store.close()

    return model_result_data

# def main():
#
#     # Take in arguments from command line
#     parser = argparse.ArgumentParser(description='Denoising Auto Encoder')
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
#     parser.add_argument('--sequence_length', type=int, default=256)
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
#     sequence_length = args.sequence_length
#
#
#     model_result_data = dae(
#         dataset_path=hd5_filepath,
#         train_building=train_building, train_start=train_start, train_end=train_end,
#         test_building=test_building, test_start=test_start, test_end=test_end,
#         meter_key=appliance,
#         sample_period=downsampling_period,
#         num_epochs=epochs,
#         sequence_length=sequence_length)
#
#     # Write options and results to file
#     with open('results/dae_json.json', 'a+') as outfile:
#         json.dump(model_result_data, outfile, sort_keys=True,
#                  indent=4, separators=(',', ': '))
#     print(model_result_data)
#
# if __name__ == "__main__":
#     main()
#
#     # python dae.py --datapath ../data/REDD/redd.h5 --train_building 1 --train_end 2011-05-10 --test_building 1 --test_start 2011-05-10 --appliance fridge --sampling_rate 20 --epochs 100 --sequence_length 256
#     # python dae.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-10 --test_building 1 --test_start 2011-05-10 --appliance fridge --sampling_rate 20 --epochs 1 --sequence_length 256
#
#     # python dae.py --datapath ../data/REDD/redd.h5 --train_building 1 --train_start 2011-03-01 --train_end 2011-05-20 --test_building 1 --test_start 2011-05-20 --test_end 2011-12-20 --appliance fridge --sampling_rate 60 --epochs 5 --sequence_length 256
