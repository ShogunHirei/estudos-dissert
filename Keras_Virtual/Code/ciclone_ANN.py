"""
File: ciclone_ANN.py
Author: ShogunHirei
Description: Uso de redes neurais para predizer componentes de velocidade em
             valor fixo de y (Slice em -0.2).
"""

import numpy as np
import re
import os
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from datetime import datetime

# Preparar input da rede neural
# Usando dados em caminho relativo

# Será utilizada a velocidade de entrada como parametro de distinção entre as
# análises

# Extraindo informações de arquivos CSV e valor da velocidade de entrada
DF = [(read_csv(dado.path), re.findall(r'_\d*_', dado.path)[0][1:-1])
      for dado in os.scandir('../Ciclone/ANN_DATA/')]

# Separando os dados de posição para X e Z (y fixo)
XZ = [dado[0][['Points:0', 'Points:2']] for dado in DF]

# Valores de velocidade com o mesmo shape dos outros inputs
INPUT_U = [[float(dado[1])]*len(XZ[0]) for dado in DF]

# Componentes de velocidade dos pontos (OUTPUT)
U_xyz = [dado[0][['U:0', 'U:1', 'U:2']] for dado in DF]

XZ = np.array([np.array(sample) for sample in XZ])
U_xyz = np.array([np.array(sample) for sample in U_xyz])
INPUT_U = np.array([np.array(sample) for sample in INPUT_U])

# Convertendo shape das velocidades para ficarem de acordo input de posição
INPUT_U = INPUT_U.reshape(XZ.shape[0], XZ.shape[1], 1)

# ETAPA DE PADRONIZAÇÃO DE DADOS
print(U_xyz[30:35, :15])
# Dados de posição são repetidos
XZ_scaler = MinMaxScaler().fit(XZ[0])
INPUT_U_scaler = MinMaxScaler().fit(INPUT_U[:, 0])
Ux_scaler = MinMaxScaler().fit(U_xyz[..., 0])


































