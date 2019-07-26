"""
Regressão Problema de cavidade
Usando Redes Neurais MLP para determinar propriedades em simulação
"""
import os
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Extração de dados dos arquivos do OpenFoam
# Extraindo o dados de velocidade do último tempo de todos os 11 exemplos

# Local do arquivo
CAVITY_FOLDER_REMOTE = r'../Cavity_Neural_Networks/'

# Regex para extrair dados numéricos
search_pattern = "\([-]?(\d*.\d*?)?(e-)?\d* [-]?(\d*.\d*)?(e-)?\d* [-]?(\d*.\d*)?\)"

NU = {}
TIME = {}

# Iterando em cada um dos exemplos de cavidade
for sample in os.scandir(CAVITY_FOLDER_REMOTE):
    # Definindo a pasta que contem o dado de transporte
    TRANSPORT_FILE_PATH = sample.path+r'/constant/transportProperties'
    TIME_FILE_PATH = max([folder.name for folder in os.scandir(sample) if re.match("[0-9.]", folder.name)])
    with open(TRANSPORT_FILE_PATH) as trans_prop:
        for line in trans_prop.readlines():
            if 'nu' in line.split():
                NU[sample.name] = [float(num[:-1]) for num in line.split()
                                   if re.match(r'(\d*.\d*);', num)][0]
    # Abrindo arquivo da velocidade do último tempo
    with open(sample.path+"/"+TIME_FILE_PATH+'/U') as U_file:
        U_VALUES = []
        for line in U_file.readlines():
            if re.match(search_pattern, line):
                U_value = (line.split(' ')[0].strip('('), line.split(' ')[1], line.split(' ')[2][:-1].strip(')'))
                U_value = tuple([float(num) for num in U_value])
                U_VALUES.append(U_value)
        TIME[sample.name+'_'+TIME_FILE_PATH] = U_VALUES

# Foram gerados dois dicionários com os dados de velocidade e Nu (Reynolds)
# Transformá-los em array com a dimensão Nª componentes * Nº amostras * Nº celulas
# DATA_ARRAY = np.array([(NU[nu], float(t.strip(nu+'_')), TIME[t][0], TIME[t][1]) for nu, t in zip(NU, TIME) ])
DATA_ARRAY = []

for t in TIME.keys():
    nu, T = t.split('_')[0], t.split('_')[1]
    for U in TIME[t]:
        DATA_ARRAY.append([NU[nu], float(T), U[0], U[1]])

DATA_ARRAY = np.array(DATA_ARRAY)
print(DATA_ARRAY.shape)

