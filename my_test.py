
"""imports for SNN Toolbox.
"""

import os
import time
import numpy as np

from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
# from keras.utils import np_utils
from tensorflow.python.keras import utils as np_utils

from snntoolbox.bin.run import main
from snntoolbox.utils.utils import import_configparser, apply_modifications

print("starting...")


# -v /home/bjoern/dev/fzi/snn_toolbox:/opt/project -v /home/bjoern/dev/fzi/ann_base_result:/opt/data/
# PYTHONPATH=/usr/bin/nest/lib/python3.7/site-packages:$PYTHONPATH

data_dir = '/opt/data/'

print("data: " + str(os.path.abspath(data_dir)))
files = os.listdir(data_dir)
for f in files:
    print(f, flush=True)
    pass


# import pyNN.nest as brain_sim

path_wd = os.path.abspath(data_dir)

import glob
list_of_files = glob.glob(path_wd+"/snapshot_*.h5") # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
model_name =  os.path.splitext(latest_file)[0]

"""
model was generated with:

optimizations = Optimizations(
    mirrored_sampling=True,
    fitness_shaping=True,
    weight_decay=True,
    discretize_actions=False,
    gradient_optimizer=False,
    observation_normalization=True,
    divide_by_stdev=False
)
"""

"""
x_test and y_test were generated with: 

    inputs=[]
    results=[]
    for _ in range(timestep_limit):
        # ...more code...
        inputs.append(ob[None])
        ac = model.predict_on_batch(ob[None])
        results.append(ac)
        ac = ac[0]
        # ...more code...
    x_test = np.concatenate(inputs)
    y_test = np.concatenate(results)
    np.savez_compressed('x_test', x_test)
    np.savez_compressed('y_test', y_test)  
"""


# SNN TOOLBOX CONFIGURATION #
#############################

# Create a config file with experimental setup for SNN Toolbox.
configparser = import_configparser()
config = configparser.ConfigParser()

config['paths'] = {
    'path_wd': path_wd,  # Path to model.
    'dataset_path': path_wd,  # Path to dataset.
    'filename_ann': model_name  # Name of input model.
}

config['tools'] = {
    'evaluate_ann': True,  # Test ANN on dataset before conversion.
    'normalize': False,
}

config['simulation'] = {
    'simulator': 'nest',  # Chooses execution backend of SNN toolbox.
    'duration': 50,  # Number of time steps to run each sample.
    'num_to_test': 5,  # How many test samples to run.
    'batch_size': 1,  # Batch size for simulation.
}

config['output'] = {
    'plot_vars': {  # Various plots (slows down simulation).
        'spiketrains',  # Leave section empty to turn off plots.
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'
    }
}

# Store config file.
config_filepath = os.path.join(path_wd, 'snn_tb_config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)

# RUN SNN TOOLBOX #
###################

"""
running in notebook gives:
TypeError: 'ObservationNormalizationLayer' object is not subscriptable

stacktrace: ... ~/snn_toolbox/snntoolbox/bin/utils.py line 91 "model_parser.parse()" ...
"""


main(config_filepath)


print("...finished")