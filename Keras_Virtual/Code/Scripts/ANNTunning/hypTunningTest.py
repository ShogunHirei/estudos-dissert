"""
File: hypTunningTest.py
Author: ShogunHirei
Description: Hyperparameters tunning test with the keras-tuner package
"""

# Python Builtins
import sys, os
import numpy as np

# Packaeg imports
import kerastuner as kt
import tensorflow.keras.optimizers as opts
from tensorflow_docs.modeling import EpochDots
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, Input, concatenate, add, Reshape, RepeatVector
from tensorflow.keras.models import Sequential, Model

# Add top-level packaeg to PATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import auxiliar_functions as af

# Dummy data to test tensorboard 
X_TRAIN = np.random.random(size=(100, 30, 4))
Y_TRAIN = np.random.random(size=(100, 30, 2))
# Changing to dicts
X_TRAIN = {f'var{i+1}': X_TRAIN[..., i] for i in range(4)}
Y_TRAIN = {f'out{i+1}': Y_TRAIN[..., i] for i in range(2)}

print(f'X_TRAIN: {type(X_TRAIN)}')
print(f'Y_TRAIN: {type(Y_TRAIN)}')

# Dummy test data for training
X_TEST = np.random.random(size=(20, 30, 4))
Y_TEST = np.random.random(size=(20, 30, 2))

# Building model
def model_building(hp):
    # Initializing the model inside function local namespace
    # model = Model()

    # Activation Function
    # ac_fn = hp.Choice('lyr_0_ac_fn', ['relu', 'tanh', 'sigmoid'])

    # Defining ANN structure
    # Input layer
    INPUT_LYR_1 = Input(shape=(30, 1), name='var1')
    INPUT_LYR_2 = Input(shape=(30, 1), name='var2')
    INPUT_LYR_3 = Input(shape=(30, 1), name='var3')
    INPUT_LYR_4 = Input(shape=(30, 1), name='var4')

    # Models inputs and outputs 
    INPUTS = [INPUT_LYR_1, INPUT_LYR_2, INPUT_LYR_3, INPUT_LYR_4]

    # Model structure definitition
    # Concatenation of coordenates
    CONCAT = concatenate(INPUTS[:3])

    #### Model has an encoder part for the compressing of input data 
    ENCD_LYRS = []

    print('Units and activation function ready!')

    #### Decided for the rate of compression (layers units reduction) of the
    #### encoder part 
    first_layer_units = 20 #hp.Int('lyr_0', 8, 32, step=4)
    compression_rate = 1 # hp.Float('comp_rate', 1.0, 2.0)

    # First layer of the encoder
    ENCD_LYRS.append(Dense(first_layer_units,
                           activation='tanh')(CONCAT)
                    )

    # Number of layers of the encoder part
    n_layers_encoder = 7 # hp.Int('enc_size', 2, 8, step=1)

    # Inittially the same ac_fn will be applied
    units = int(first_layer_units/compression_rate) + 1
    for i in range(n_layers_encoder):
        ENCD_LYRS.append(Dense(units, activation='tanh')(ENCD_LYRS[i]))
        units = int(units * compression_rate)
    print('Encoder ready!')

    # After the encoder, it is inserted the last variable (Inlet/Re)
    CONCAT_2 = concatenate([INPUTS[-1], ENCD_LYRS[-1]])

    print('Inserted Input!')

    # Decoder part number of layers
    n_layers_decoder = 8 # hp.Int('dec_size', 2, 8)

    ## ADDITION OF RESIDUAL CONNECTIONS!
    # First, the residual connections will be between the layers in the 
    # decoder and the encoder.
    # Encoder layers index
    enc_lyrs_idx = hp.Choice('enc_idx', list(range(1,n_layers_encoder)))

    # Decoder layer index
    dcd_lyrs_idx = hp.Choice('dcd_idx', list(range(1,n_layers_decoder)))

    # With the idx the layer in the decoder will be the add layer 
    # which will call the encoder layer
    print(f"The choosen ENC_IDX: {enc_lyrs_idx} | DCD_IDX: {dcd_lyrs_idx}")

    DCD_LYRS = [CONCAT_2]
    for i in range(n_layers_decoder):
        # The first option is to modify the predecessor to be compatible with 
        # encoder layer shape
        if i == int(dcd_lyrs_idx):
            # The connection between the layers 
            DCD_LYRS.append(add([DCD_LYRS[i],
                                 ENCD_LYRS[enc_lyrs_idx]]))

        # To the tensor shape be the same to the residual connection to work properly
        elif i == int(dcd_lyrs_idx - 1):
            # Number of neurons of the specific layer in the connection
            ENCD_UNITS = int(ENCD_LYRS[enc_lyrs_idx].get_shape()[-1])
            # The number of units in the predecessor layer be the same of the
            # encoder layer choosen by the algorithm
            DCD_LYRS.append(Dense(ENCD_UNITS, 
                                  activation = 'tanh'
                                  # activation=hp.Choice('lyr_{i}_ac_fn',
                                                       # ['relu', 'tanh', 'sigmoid'])
                                  )(DCD_LYRS[i]))
        else:
            # Changing the `decoder` section activation function
            DCD_LYRS.append(Dense(20, #hp.Int(f'dec_unit_{i}', 5, 10), 
                                  activation = 'tanh'
                                  # activation=hp.Choice('lyr_{i}_ac_fn',
                                                       # ['relu', 'tanh', 'sigmoid'])
                                  )(DCD_LYRS[i]))

    print('Decoder ready!')
    # Defining Output layers 
    OUT1 = Dense(1, activation='tanh', name='out1')(DCD_LYRS[-1])
    OUT2 = Dense(1, activation='tanh', name='out2')(DCD_LYRS[-1])
    OUTPUTS = [OUT1, OUT2]

    model = Model(inputs=INPUTS, outputs=OUTPUTS)

    # Compiling: Optimizers and loss functions configuration
    # LOSS_FUNCTIONS = [hp.Choice('loss_fn', ['mse','mae']), hp.Choice('loss_fn', ['mse','mae'])]
    LOSS_FUNCTIONS = ['mse', 'mae']
    LOSS_WEIGHTS = [1, 0.5]

    # ch_optimizer = hp.Choice('optimizers', ['adam', 'nadam', 'RMSprop'])
    ch_optimizer = 'adam'

    # Compiling..
    model.compile(ch_optimizer,
                  loss=LOSS_FUNCTIONS,
                  loss_weights=LOSS_WEIGHTS,
                 )
                  # metrics = ['lr'])

    return model


tuner = kt.tuners.hyperband.Hyperband(model_building,
                                      objective='val_loss',
                                      max_epochs=10,
                                      factor=5,
                                      executions_per_trial = 1,
                                      directory = './tunning_test/',
                                      project_name = 'KerasTunerTesting',
                                      overwrite = True
                                      )

# Defining callbacks
CBCK =[
        TensorBoard(log_dir='./tunning_test/tensorboard/'),
        EarlyStopping(patience=5),
        EpochDots(report_every=20)
        ]

# Print(?) space summary
# tuner.search_space_summary()
print("Starting search")

tuner.search(X_TRAIN, Y_TRAIN,  
             epochs=15,
             validation_split=0.2,
             callbacks=CBCK)

# Retrieving the best model searched
best_models = tuner.get_best_models(num_models=1)

# saving best model 
best_models[0].save('./best_models.h5')

# Printing summary results
tuner.results_summary()






    

