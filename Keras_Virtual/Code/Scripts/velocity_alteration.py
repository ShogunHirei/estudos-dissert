"""
File: velocity_alteration.py
Author: ShogunHirei
Description: Script para mudar valores de velocidades de entrada nos arquivos
             de simulação do ciclone. Remover tempo de 4200 anterior e mudar
             velocidade em determinado padrão
"""

import sys
import os
import re

# Considerar que script será realizado dentro da pasta do caso
FOLDER = sys.argv[1]

# Considerando que valor de velocidade será inserido como argumento
VELOC_VALUE = sys.argv[2]

# Obter listas das pastas para as iterações de passo de tempo
TIME_FOLDERS = []
for entry in os.scandir(FOLDER):
    if len(re.findall(r"[\D]", entry.name))>1:
        continue
    else:
        TIME_FOLDERS.append(entry)

print(TIME_FOLDERS)

# Removendo tempo de 4200 ('estado estacionário') da lista de pasta para
# executar as alterações de velocidades nas outras pastas de tempo
for entry in TIME_FOLDERS:
    if entry.name == '4200':
        TIME_FOLDERS.remove(entry)
print(TIME_FOLDERS)
#os.system('rm ./4200 -r')
# Realizar alterações nos arquivos de velocidade para todos os tempos
for time in TIME_FOLDERS:
    for prop in os.scandir(time.path):
        if prop.name == 'U':
            U_PATH = prop.path
            with open(U_PATH, 'r+') as FILE:
                for line in FILE.readlines():
                    if 'refValue' in line.split():
                        old_value = [data for data in line.split()
                                     if re.match(r'-\d*.\d*;', data)][0][:-1]
                        OLD_string = f'        refValue        uniform {old_value};'
                        NEW_string = f'        refValue        uniform -{VELOC_VALUE};'
                        os.system(f'sed -i "s/{OLD_string}/{NEW_string}/" {U_PATH}')

