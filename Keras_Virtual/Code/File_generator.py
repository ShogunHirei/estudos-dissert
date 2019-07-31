"""
Usar modelo de rede neural gerado para escrever
em formato interpretável pelo OpenFoam
"""

import sys
import numpy as np
from keras.models import load_model


# ETAPAS;
# -> Carregar modelo 
# -> Usar como dados de entrada NU e (x, y)
# -> Recuperar dados da rede
# -> Reescalar dados para as variáveis normais
# -> Abrir arquivo para escrita 
# -> Escrever dados no formato (ux uy 0)

######## CUIDADO COM OS VALORES PARA A RENORMALIZAÇÃO ############
R = [0.02500000000000032, 9.5, 9.5, 0.8515943956457772, 0.4965164834260455]
xMED = [0.03499999999999968, 9.5, 10.5, 0.0008716043542227244, 0.0001354834260454543]

# Pegar nome do arquivo de modelo de rede neural a partir da linha de comando
FILENAME = sys.argv[1]
MODEL = load_model(FILENAME)

# Gerar input de dados
NU = 0.01
INPUT = []

# como os dados foram normalizados antes da inserção no modelo
# eles devem ser escalados para serem interpretados corretamente
norm = lambda x, i: (x - xMED[i])/(R[i])
re_norm = lambda x, i: x*R[i] + xMED[i]
NU = norm(NU, 0)

for i in range(20):
    for j in range(20):
        INPUT.append([NU, norm(i, 1), norm(j, 2)])

INPUT = np.array(INPUT).reshape(-1, 400, 3)

PREDICTIONS = MODEL.predict(INPUT)

RE_NORM_PREDICTIONs = []

for data in PREDICTIONS[0]:
    RE_NORM_PREDICTIONs.append([re_norm(data[0], 3), re_norm(data[1], 4)])

print(RE_NORM_PREDICTIONs.__len__())

with open('./Generated Files/cavity_MLP_U', 'w') as results:
    for linha in RE_NORM_PREDICTIONs:
        results.write(f'({linha[0]:.6f} {linha[1]:.6f} 0)\n')

print('Finished!')


