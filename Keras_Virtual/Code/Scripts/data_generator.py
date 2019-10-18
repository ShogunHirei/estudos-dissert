"""
File: data_generator.py
Author: ShogunHirei
Description: Script para gerar arquivo csv a partir do modelo de rede salvo
"""

import os
import sys
import numpy as np
from joblib import load
from keras.models import load_model
from pandas import read_csv, concat, DataFrame

# Arquivo de entrada será da rede utilizada
FILE_MODEL = sys.argv[1]

# Velocidade de entrada a ser prevista
VEL_ENTR = sys.argv[2]

# Local no qual será salvo o arquivo gerado
FOLDER = sys.argv[3]

# Gerar dado para predição (utilizando o número de pontos máximo)
VEL_ARR = np.array([[VEL_ENTR]*868]).reshape(-1, 1)

# Usando conjunto de pontos do SLICE original
XZ = read_csv(os.scandir('../../Ciclone/ANN_DATA').__next__().path)[['Points:0', 'Points:2']]

# Carregando parametros de padronização para input e interpretação dos dados
SC_DIR = '../Models/Multi_Input/Scaler/'
XZ_scaler = load(SC_DIR+'points_scaler.joblib')
InputU_scaler = load(SC_DIR+'U_input_scaler.joblib')
Ux_scaler = load(SC_DIR+'Ux_scaler.joblib')
Uy_scaler = load(SC_DIR+'Uy_scaler.joblib')
Uz_scaler = load(SC_DIR+'Uz_scaler.joblib')

# Transformando os dados de entrada
XZ = XZ_scaler.transform(XZ).reshape(1, -1, 2)
VEL_ARR = InputU_scaler.transform(VEL_ARR).reshape(1, -1, 1)

# Carregando o modelo de rede neural para previsão
model = load_model(FILE_MODEL)

# Previsão de valores de entrada
PREDIC = model.predict({'XZ_input': XZ, 'U_entr': VEL_ARR})

# Retornando os dados para a escala anterior
Ux = DataFrame(Ux_scaler.inverse_transform(PREDIC[..., 0]).reshape(-1), columns=['U:0'])
Uy = DataFrame(Uy_scaler.inverse_transform(PREDIC[..., 1]).reshape(-1), columns=['U:1'])
Uz = DataFrame(Uz_scaler.inverse_transform(PREDIC[..., 2]).reshape(-1), columns=['U:2'])

# Inserindo valor dos pontos de Y
XYZ = read_csv(os.scandir('../../Ciclone/ANN_DATA/').__next__().path)[['Points:0', 'Points:1', 'Points:2']]

# Geração de arquivo .CSV para leitura
FILENAME = f'NEW_SLICE_{VEL_ENTR}.csv'

SLICE_DATA = concat([Ux, Uy, Uz, XYZ], sort=True, axis=1)

# Escrevendo o header no formato do paraview
with open(FOLDER+FILENAME, 'w') as filename:
    HEADER = ''
    for col in list(SLICE_DATA.columns):
        HEADER += '\"' + col + '\",'
    filename.write(HEADER[:-1])
    filename.write('\n')

SLICE_DATA.to_csv(FOLDER + FILENAME, index=False, header=False, mode='a')
print("Finished Writing!")

