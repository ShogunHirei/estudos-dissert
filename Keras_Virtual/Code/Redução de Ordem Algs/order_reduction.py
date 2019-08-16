"""
Script para realizar a redução de ordem dos dados gerados

Primeiro utilizando a POD para realizar a decomposição
    dos espaço vetorial de dados
"""

import mordred as md
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense

# Obtendo dados para manipulação
DF = read_csv('./DATA_FOLDER/cavity_U_0.1')

# Gerar numpy arrays no formato [ NU, [ [Ux, Uy], [Ux, Uy] ... ] ]
DATA_ARRAY = []

# for nu in set(DF['NU']):
    # DATA_ARRAY



