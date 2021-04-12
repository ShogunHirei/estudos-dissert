"""
File: training_reload.py
Author: ShogunHirei
Description: Script for restart training of model with newer parameters
"""

import os, sys, argparse, datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from Scripts.auxiliar_functions import TrainingData, tf_less_verbose
from pandas import read_csv

# Parsing arguments
parser = argparse.ArgumentParser()

# Case folder 
parser.add_argument("samples_path", help="Data folder path", )

# Model Path
parser.add_argument('model_path', help="The path for the model to train", )

# Scaler Folder
parser.add_argument('-sf', '--scaler_folder', help="""Scaler path to load scalers of  
                    variables, defaults to the `model_path` folder, if fails 
                    changes to current working folder""") 

# Base fold
parser.add_argument("-bf", "--base_fold", help="""The folder where will be stored
                    the information, defaults to `model_path` base folder""")

# Variables to write
parser.add_argument('--vars',  nargs='+', default=['U'], help="""The Variables 
                    which will be added to the generate .CSV and will have the 
                    OF variable file generated""")

# To enable verbose from Tensorflow warnings
# If it is not specified the output will be supressed
parser.add_argument('-v', '--verbose', action='store_true', help='''Supress TF 
                    warnings\nIf it is not specified the output will be supressed''')

# If unidimensional Reynods number
parser.add_argument('-R1', '--Reynolds', action='store_false', 
                    help='''Unidimensional Reynods, if defined Reynolds will not
                         be unidimensional''')

# Parsing arguments
args = parser.parse_args()

def main(args):
    # Loading  path for model and directories
    MODEL_PATH = args.model_path
    DATA_FOLD = args.samples_path
    VARS = args.vars

    if args.base_fold:
        MOD_FOLD = args.base_fold
    else:
        MOD_FOLD = os.path.dirname(MODEL_PATH) + '/'

    if args.scaler_folder:
        SCALER_FOLDER = args.scaler_folder + '/'
        LOAD = True
    elif os.path.exists(MOD_FOLD + 'Scalers/'):
        SCALER_FOLDER = MOD_FOLD + 'Scalers/'
        LOAD = bool(input(f'Load scalers from {SCALER_FOLDER}? '))
    else:
        SCALER_FOLDER = './Scalers/'
        LOAD = bool(input(f'Load scalers from {SCALER_FOLDER}? '))

    # Working Folder
    NEWT_DIR = MOD_FOLD+'/NEW_TRAINING/' 
    
    # In the model folder, make a new past for new training
    if not os.path.exists(NEWT_DIR):
        os.mkdir(NEWT_DIR)
    else:
        now = datetime.datetime.now().strftime("%d%m-%H%M")
        NEWT_DIR = MOD_FOLD+f'/NEW_TRAINING_{now}/' 
        os.mkdir(NEWT_DIR)
    
    
    # Getting Model for continuing training
    try:
        # The model maybe has custom objects, so if this is not included 
        #  the model will not be loaded
        model = load_model(MODEL_PATH)
        print("Sucess loading the model!")
    except:
        print("Error reading model!\n\nCheck for path or TF CustomObjects!")
    
    
    
    # Data generation for new training
    DATA = TrainingData(DATA_FOLD, FACTOR=5283.80102)
    DATA.save_dir = NEWT_DIR
    DATA.scaler_folder = SCALER_FOLDER
    DATA.info_folder = MOD_FOLD+'/Info'
    
    
    # Loading Data for Training
    print("=".rjust(35,'=') + ' TRAINING ' + '='.ljust(35, '='))
    # There is no need for calculating the scalers again, because 
    #   the data is the same
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.20,
                                                     inp_labels=['Points', 'Inlet'],
                                                     out_labels=VARS,
                                                     mag=[],
                                                     load_sc=LOAD)
    
    # Creating dicts for inputs
    X = DATA.training_dict(X_TRAIN, 0)
    Y = DATA.training_dict(Y_TRAIN, 1)
    # Para criar os dicionários dedados de validação 
    V_X = DATA.training_dict(X_TEST, 0)
    V_Y = DATA.training_dict(Y_TEST, 1)
    
    if args.reynolds:
        # Check for dimension of Inlet
        if Re_un:
            X['Inlet_U'] = X['Inlet_U'][:, 0, :]
            V_X['Inlet_U'] = V_X['Inlet_U'][:, 0, :]
            print('Loaded unidimensional Re for input')
    
    opt = Adam()
    
    model.compile(optimizer=opt,
                  # loss={'U_0':'mae','U_1':'mae','U_2':'mae','p':'mae'
                  loss={'p':'mae',
                        'Res_0':'mse','Res_1':'mse','Res_2':'mse'},
                  # loss_weights={'U_0':0.7,'U_1':0.7,'U_2':0.7, 'p':0.9
                  loss_weights={'p':0.9,
                                'Res_0':0.3,'Res_1':0.3,'Res_2':0.3,})
    
    CBCK = DATA.list_callbacks(NEWT_DIR, monit='val_loss')
    
    # Add Callback for saving models checkpoint
    MDCHPT = ModelCheckpoint(NEWT_DIR+'model_checkpoint.h5', monitor='val_loss', 
                             save_freq=200, save_best_only=True, verbose=2)
    CBCK.append(MDCHPT)
    
    # Running training
    hist = model.fit(X,Y,validation_data=(V_X, V_Y),
                     batch_size=32, epochs=5000, callbacks=CBCK)
    
    print("Salvando modelo para futuro treinamento")
    model.save(NEWT_DIR+f'model_trained.h5')
    
    return None



if __name__ == '__main__':
    if not args.verborse:
        tf_less_verbose()

    # Running...
    main(args)
    print("Finished!")
