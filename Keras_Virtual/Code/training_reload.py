"""
File: training_reload.py
Author: ShogunHirei
Description: Script for restart training of model with newer parameters
"""

import os, sys
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, SGD
from Scripts.auxiliar_functions import TrainingData, make_folder
from pandas import read_csv

# Loading  path for model and directories
MOD_FOLD = sys.argv[1]
DATA_FOLD = sys.argv[2]
EVAL_CASE = sys.argv[3]
try:
    LOAD = sys.argv[4]
except:
    LOAD = False

# Working Folder
NEWT_DIR = MOD_FOLD+'/NEW_TRAINING/' 

# Getting Model for continuing training
try:
    # The model maybe has custom objects, so if this is not included 
    #  the model will not be loaded
    model = load_model(MOD_FOLD+[p for p in os.listdir(MOD_FOLD) if p[:8]=='model_tr'][0])
    print("Sucess loading the model!")
except:
    print("Error reading model!\n\nCheck for path or TF CustomObjects!")


# In the model folder, make a new past for new training
os.mkdir(NEWT_DIR)

# Data generation for new training
DATA = TrainingData(DATA_FOLD, FACTOR=5283.80102)
DATA.save_dir = NEWT_DIR
DATA.scaler_folder = MOD_FOLD+'/Scalers'
DATA.info_folder = MOD_FOLD+'/Info'


# Loading Data for Training
print("=".rjust(35,'=') + ' TRAINING ' + '='.ljust(35, '='))
# There is no need for calculating the scalers again, because 
#   the data is the same
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.20,
                                                 inp_labels=['Points', 'Inlet'],
                                                 out_labels=['U', 'p', 'Res'],
                                                 mag=[],
                                                 load_sc=LOAD)

# Creating dicts for inputs
X = DATA.training_dict(X_TRAIN, 0)
Y = DATA.training_dict(Y_TRAIN, 1)
# Para criar os dicionários dedados de validação 
V_X = DATA.training_dict(X_TEST, 0)
V_Y = DATA.training_dict(Y_TEST, 1)

opt = Opts.Adam()

model.compile(optimizer=opt,
              loss={'U_0':'mae','U_1':'mae','U_2':'mae','p':'mae'
                    'Res_0':'mse','Res_1':'mse','Res_2':'mse'},
              loss_weights={'U_0':0.7,'U_1':0.7,'U_2':0.7, 'p':0.9
                            'Res_0':0.3,'Res_1':0.3,'Res_2':0.3,})

CBCK = DATA.list_callbacks(NEWT_DIR)

# Add Callback for saving models checkpoint
MDCHPT = ModelCheckpoint(NEWT_DIR+'model_checkpoint.h5', monitor='val_loss', period=200,
                         save_best_only=True, verbose=2)
CBCK.append(MDCHPT)

# Running training
hist = model.fit(X,Y,validation_data=(V_X, V_Y),
                 batch_size=32, epochs=5000, callbacks=CBCK)

print("Salvando modelo para futuro treinamento")
model.save(NEWT_DIR+f'model_trained.h5')

# For Testing the trained Model in data for Test
# Creating the folder of file 
CASE = '/DATA_U_10.00/'
os.mkdir(NEWT_DIR+CASE)
# Loading data for case
TEST = TrainingData(EVAL_CASE, FACTOR=5283.80102)
# Pointing scalers folders 
TEST.scaler_folder = DATA.scaler_folder
# Generating Data for evaluation of model output
# Ignoring the original outputs data
X, _ = TEST.data_gen(EVAL=True, out_labels=['U', 'p'])
# Ordering data for Inlet_U for the net with skip_connection
DX = TEST.training_dict(X, 0)
temp = DX['Inlet_U'][:,0,:]
DX['Inlet_U'] = temp
# Predicting DATA for model and saving data in FOLDER
TEST.predict_data_generator(model, DX, NEWT_DIR+ CASE + 'result_pred.csv') ;

# To Generate a file of OpenFOAM variable
RESULT = read_csv(NEWT_DIR+ CASE + '/result_pred.csv');
TEST.U_for_OpenFOAM(RESULT[['U:0', 'U:1', 'U:2']], FOLD+CASE+'ANN_Vel')
TEST.U_for_OpenFOAM(RESULT['p'], FOLD+CASE+'ANN_press', VECTOR='False')
