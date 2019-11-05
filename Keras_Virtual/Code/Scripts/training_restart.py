"""
File: training_restart.py
Author: ShogunHirei
Description: Pequeno script para ser copiado ao fim de cada treinamento para
             sua continuação, caso necessário.
"""

import re
import os
import sys
import numpy as np
from .auxiliar_functions import TrainingData
from keras.models import load_model
from keras.callbacks import EarlyStopping, Tensorboard, ReduceLROnPlateau
from keras.optimizers import Nadam


# Usando parametros de entrada para generalizar script
MODEL_FN = sys.argv[1]

SCALER_DIR = sys.argv[2]

U_MAG = True if sys.argv[3] == 'True' else False


# Carregando dados de treinamento
DATA = TrainingData(U_mag=U_MAG, scaler_dir=SCALER_DIR)

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = DATA.data_gen(test_split=0.25, U_mag=U_MAG)

# Continuando o treinamento de determinado
model = load_model(MODEL_FN,)

# CALLBACKS
TB = Tensorboard(logdir='./', histogram_freq=90)
ES = EarlyStopping(monitor='loss', min_delta=1E-5, patience=350,
                   restore_best_weights=True)
RLRP = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=100, verbose=1,
                         min_lr=1E-10)
CBCK = [TB, ES, RLRP]

print('Inicio de treinamento')

# X_TRAIN.shape[0] é a quantidade de amostras
model.fit({'XZ_input': X_TRAIN[..., :2],
           'U_entr': X_TRAIN[..., 2].reshape(X_TRAIN.shape[0], -1, 1)},
          {'Ux_Output': Y_TRAIN[..., 0].reshape(Y_TRAIN.shape[0], -1, 1),
           'Uy_Output': Y_TRAIN[..., 1].reshape(Y_TRAIN.shape[0], -1, 1),
           'Uz_Output': Y_TRAIN[..., 2].reshape(Y_TRAIN.shape[0], -1, 1),
           'Mag': Y_TRAIN[..., 3].reshape(Y_TRAIN.shape[0], -1, 1)},
          validation_data=({'XZ_input': X_TEST[..., :2],
                            'U_entr': X_TEST[..., 2].reshape(X_TEST.shape[0], -1, 1)},
                           {'Ux_Output': Y_TEST[..., 0].reshape(Y_TEST.shape[0], -1, 1),
                            'Uy_Output': Y_TEST[..., 1].reshape(Y_TEST.shape[0], -1, 1),
                            'Uz_Output': Y_TEST[..., 2].reshape(Y_TEST.shape[0], -1, 1),
                            'Mag': Y_TEST[..., 3].reshape(Y_TEST.shape[0], -1, 1)},
                           ),
          epochs=10000, batch_size=4, callbacks=CBCK, verbose=0)
print("Finished Trainning U")

