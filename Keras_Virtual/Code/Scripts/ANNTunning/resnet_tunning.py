
"""
File: resnet_tunning.py
Author: ShogunHirei
Description: Residual Neural Network Hyperparameters tunning 
	     with the keras-tuner package
"""

# Python Builtins
import sys, os
import numpy as np

# Packaeg imports
import kerastuner as kt
import tensorflow.keras.optimizers as opts
#from tensorflow_docs.modeling import EpochDots
#from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, Input, concatenate, add, Reshape, RepeatVector
from tensorflow.keras.models import Sequential, Model

# Add top-level packaeg to PATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from auxiliar_functions import *

# Carregando dados para Treinamento
ANN_FOLDER = sys.argv[1]
FOLD = sys.argv[2]
try:
    IN_SCALER_FOLD = sys.argv[3]
except:
    IN_SCALER_FOLD = False

             
# Criando diretório para operações de gravação
# Gerando pastas para armazenar os dados
BASE_DIR, INFO_DIR, SCALER_FOLDER = make_folder(FOLD)

# Geração de Conjunto de treinamento e teste
DATA = TrainingData(ANN_FOLDER, FACTOR=5283.80102)
DATA.save_dir = BASE_DIR
DATA.info_folder = INFO_DIR

# To avoid repetitive scaler generation
# If it was inserted a path to the scalers it will use it 
if IN_SCALER_FOLD:
    DATA.scaler_folder = IN_SCALER_FOLD
    LOAD_SC = True
    print('Ok! I\'ll load the scalers from ', IN_SCALER_FOLD)
else:
    DATA.scaler_folder = SCALER_FOLDER
    LOAD_SC = False
    print('Right! Create and write scalers in ', SCALER_FOLDER)

# Training Data
print("=".rjust(35,'=') + ' TRAINING ' + '='.ljust(35, '='))
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.20,
                                                 inp_labels=['Points', 'Inlet'],
                                                 out_labels=['U', 'Res',],
                                                 mag=[],
                                                 load_sc=LOAD_SC,)

# Creating dicts for inputs
X = DATA.training_dict(X_TRAIN, 0)
Y = DATA.training_dict(Y_TRAIN, 1)
# Creating the validation set dictionaries
V_X = DATA.training_dict(X_TEST, 0)
V_Y = DATA.training_dict(Y_TEST, 1)

# Reshaping data for inlet_U
X['Inlet_U'] = X['Inlet_U'][:, 0, 0]
V_X['Inlet_U'] = V_X['Inlet_U'][:, 0, 0]

CBCK = DATA.list_callbacks('./tunning_test/Res_Net_Tunning/tensorboard/', monit='val_loss')


# Building model
def model_building(hp):
    # Activation Function
    # ac_fn = hp.Choice('lyr_0_ac_fn', ['relu', 'tanh', 'sigmoid'])

    # Defining ANN structure
    # Input layer
    net_input = []
    for key in X.keys():
        if key != 'Inlet_U':
            net_input.append(Input(shape=X[key].shape[1:], dtype='float32', name=key))
        else:
    	# para a entrada da velocidade associada ao decoder
    	    B = Input(shape=(1,), dtype='float32', name='Inlet_U')

    # Model structure definitition
    # Concatenation of coordenates
    CONCAT = concatenate(net_input)

    #### Model has an encoder part for the compressing of input data 
    ENCD_LYRS = []

    print('Units and activation function ready!')

    #### Decided for the rate of compression (layers units reduction) of the
    #### encoder part 
    first_layer_units = hp.Int('lyr_0', 8, 64, step=24)
    compression_rate = hp.Float('comp_rate', 1.0, 2.0)

    # First layer of the encoder
    ENCD_LYRS.append(Dense(first_layer_units,
                           activation='tanh')(CONCAT)
                    )

    # Number of layers of the encoder part
    n_layers_encoder = hp.Int('enc_size', 2, 8, step=2)

    # Inittially the same ac_fn will be applied
    units = int(first_layer_units/compression_rate) + 1
    for i in range(n_layers_encoder):
        ENCD_LYRS.append(Dense(units, activation='tanh')(ENCD_LYRS[i]))
        units = int(units / compression_rate + 1)
    print('Encoder ready!')

    # After the encoder, it is inserted the last variable (Inlet/Re)
    # Velocity insertion
    b = RepeatVector(int(net_input[0].shape[-2]))(B) 
    CONCAT_2 = concatenate([b, ENCD_LYRS[-1]])

    print('Inserted Input!')

    # Decoder part number of layers
    n_layers_decoder = hp.Int('dec_size', 2, 8, step=2)

    ## ADDITION OF RESIDUAL CONNECTIONS!
    # First, the residual connections will be between the layers in the 
    # decoder and the encoder.
    # Encoder layers index
    enc_lyrs_idx = hp.Choice('enc_idx', list(range(1,n_layers_encoder)),
                             ordered=False)

    # Decoder layer index
    dcd_lyrs_idx = hp.Choice('dcd_idx', list(range(1,n_layers_decoder)),
                             ordered=False)

    # With the idx the layer in the decoder will be the add layer 
    # which will call the encoder layer
    #print(f"The choosen ENC_IDX: {enc_lyrs_idx} | DCD_IDX: {dcd_lyrs_idx}")

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
            DCD_LYRS.append(Dense(hp.Int(f'dec_unit_{i}', 5, 10), 
                                  activation = 'tanh'
                                  # activation=hp.Choice('lyr_{i}_ac_fn',
                                                       # ['relu', 'tanh', 'sigmoid'])
                                  )(DCD_LYRS[i]))

    print('Decoder ready!')


    # Defining Output layers 
    out_put = []
    for key in Y.keys():
        out_put.append(Dense(1, activation='tanh', name=key)(DCD_LYRS[-1]))

    model = Model(inputs=net_input + [B], outputs=out_put)

    # Compiling: Optimizers and loss functions configuration
    LOSS_FUNCTIONS = {'U_0':'mae','U_1':'mae','U_2':'mae',
                      'Res_0':'mse','Res_1':'mse','Res_2':'mse',}
    LOSS_WEIGHTS = {'U_0':0.9,'U_1':0.9,'U_2':0.9, 'Res_0':0.3,'Res_1':0.3,'Res_2':0.3,}

    ch_optimizer = 'adam'

    # Compiling..
    model.compile(ch_optimizer,
                  loss=LOSS_FUNCTIONS,
                  loss_weights=LOSS_WEIGHTS,)

    # To print the number of neurons in each layer
    lyer_n_e = [int(lyr.get_shape()[-1]) for lyr in ENCD_LYRS[1:] ]
    lyer_n_d = [int(lyr.get_shape()[-1]) for lyr in DCD_LYRS[1:] ]
    print('Model Compiled Succesfully!', f'ENC_layers: {n_layers_encoder}',
          f'DCD_layers: {n_layers_decoder}',
          f'Initial Layer: {first_layer_units} units',
          f'Encoder layer number: {lyer_n_e}',
          f'Decoder layer number: {lyer_n_d}',
          f'ENC_IDX/DCD_IDX: {enc_lyrs_idx}/{dcd_lyrs_idx}', sep='\n')

    return model


tuner = kt.tuners.hyperband.Hyperband(model_building,
                                      objective='val_loss',
                                      max_epochs=150,
                                      factor=5,
                                      executions_per_trial = 5,
                                      directory = './tunning_test/Res_Net_Tunning/',
                                      project_name = 'ResNet_U_KTuner',
                                      max_model_size = 9E6,
                                      overwrite = False
                                      )

# Print(?) space summary
# tuner.search_space_summary()
print("Starting search")

tuner.search(X, Y, validation_data=(V_X, V_Y),
             epochs=3500, batch_size=32,
             #validation_split=0.2,
             callbacks=CBCK)

# Retrieving the best model searched
best_models = tuner.get_best_models(num_models=5)

# saving best model in the base folder
for idx, mod in enumerate(best_models):
    mod.save(f'{BASE_DIR}/best_model_{idx}.h5')

print('The five best models were saved!')

# Printing summary results
tuner.results_summary()






    

