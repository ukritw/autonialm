from flask import Flask, render_template, request, redirect, url_for
from bayesian_optimization import automl_hyperopt

app = Flask(__name__)

@app.route('/')
def render_automl_form():
    return render_template('automl_input.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():

    dict = request.form
    for key in dict:

        print (key,": ",dict[key])

    args_hd5_filepath = request.form['hd5_filepath']
    args_train_building = int(request.form['train_building'])
    args_train_start = None if request.form['train_start'] == "" else request.form['train_start']
    args_train_end = request.form['train_end']
    args_test_building = int(request.form['test_building'])
    args_test_start = request.form['test_start']
    args_test_end = None if request.form['test_end'] == "" else request.form['test_end']
    args_appliance = request.form['appliance']
    args_downsampling_period = int(request.form['downsampling_period'])
    args_max_evals = int(request.form['max_evals'])
    args_metrics_to_optimize = request.form['metrics_to_optimize']
    args_num_epochs = int(request.form['num_epochs'])
    args_patience = int(request.form['patience'])

    automl_hyperopt.main(args_hd5_filepath ,args_train_building, args_train_start, args_train_end, args_test_building, args_test_start ,args_test_end,
                         args_appliance, args_downsampling_period, args_max_evals, args_metrics_to_optimize, args_num_epochs, args_patience)

    return redirect(url_for('display_results'))

@app.route('/display_result')
def display_results():

    # TODO: load json file to dict
    import json
    from pprint import pprint

    with open('bayesian_optimization/results/latest_metadata.json') as f:
        metadata = json.load(f)

    with open('bayesian_optimization/results/latest_results.json') as d:
        data = json.load(d)

    # TODO: sort json file by loss
    from operator import itemgetter
    sorted_data = sorted(data, key=itemgetter('loss'), reverse=False)

    # Change key name in Metadata
    metadata["best algorithm"] = metadata.pop("space")
    metadata["best loss"] = metadata.pop("loss")

    return render_template('display_result.html', metadata=metadata, results=sorted_data)
