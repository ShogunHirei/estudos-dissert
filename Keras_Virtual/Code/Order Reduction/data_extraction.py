"""
Reorganização do código para uso das redes neurais para
previsão de perfis de escoamento

Script para a extração de dados de determinada propriedade de um caso de estudo

Os dados da propriedade serão extraídos de arquivo fornecidos
    por nome na linha de comando
"""
import os
import re
import numpy as np
import argparse as ap

PARSER = ap.ArgumentParser(description="Automatizar extração de dados."
                           "Definindo o caminho de extração e a propriedade")
PARSER.add_argument('--filename', nargs='*', help="Path to Extract")
PARSER.add_argument('--prop', nargs='*', help="Properties")
PARSER.add_argument('--output', nargs='*', help="Output File Path")
ARGS = vars(PARSER.parse_args())

# Extração de dados dos arquivos do OpenFoam
# Extraindo o dados de velocidade do último tempo de todos os 11 exemplos

# Local do arquivo
CAVITY_FOLDER_REMOTE = ARGS['filename'][0]
PROP = ARGS['prop'][0]
OUTPUTPATH = ARGS['output'][0]

# Regex para extrair dados numéricos
search_pattern = "\([-]?(\d*.\d*?)?(e-)?\d* [-]?(\d*.\d*?)?(e-)?\d* [-]?(\d*.\d*?)?(e-)?\d*\)"

NU = {}
TIME = {}

# Iterando em cada um dos exemplos de cavidade
for sample in os.scandir(CAVITY_FOLDER_REMOTE):
    # Definindo a pasta que contem o dado de transporte
    TRANSPORT_FILE_PATH = sample.path + r'/constant/transportProperties'
    # Usando regex para pegar o amior passo de tempo do caso estudado
    TIME_FILE_PATH = max([
        folder.name for folder in os.scandir(sample)

        if re.match("[0-9.]", folder.name)
    ])
    print(f"Nu_path: {TRANSPORT_FILE_PATH}, Time_path: {TIME_FILE_PATH}")
    with open(TRANSPORT_FILE_PATH) as trans_prop:
        for line in trans_prop.readlines():
            if 'nu' in line.split():
                NU[sample.name] = [
                    float(num[:-1]) for num in line.split()

                    if re.match(r'(\d*.\d*);', num)
                ][0]
    # Abrindo arquivo da velocidade do último tempo
    with open(sample.path + "/" + TIME_FILE_PATH + '/' + ARGS['prop'][0][0]) as U_file:
        U_VALUES = []
        # Adicionando a posição da célula como informação extra
        # para que a rede neural possa avaliar a posição da
        # célula na malha
        U_POSIX = 0
        U_POSIY = 0.1/20

        for line in U_file.readlines():
            # A avaliação pelo índice da célula não foi suficiente
            # ao invés do ID, vou usar a posiçãoda célula em uma
            # maneira cartesiana (Ux, Uy)

            if re.match(search_pattern, line):
                if round(U_POSIX, 5) >= 0.1:
                    U_POSIY += round(0.1/20, 5)
                    U_POSIX = 0
                U_POSIX += round(0.1/20, 5)
                U_value = (U_POSIX, U_POSIY, line.split(' ')[0].strip('('),
                           line.split(' ')[1],
                           line.split(' ')[2][:-1].strip(')'))
                U_value = tuple([float(num) for num in U_value])
                U_VALUES.append(U_value)
        TIME[sample.name + '_' + TIME_FILE_PATH] = U_VALUES

# Foram gerados dois dicionários com os dados de velocidade e Nu (Reynolds)
# Transformá-los em array de (NU, X, Y, Ux, Uy)

DATA_ARRAY = []

for t in TIME.keys():
    nu, T = t.split('_')[0], t.split('_')[1]

    for U in TIME[t]:
        DATA_ARRAY.append([NU[nu], U[0], U[1], U[2], U[3]])

DATA_ARRAY = np.array(DATA_ARRAY)
print(DATA_ARRAY.shape)

# Escrever dados para arquivo tipo .CSV para facilitar leitura em próxima etapa

with open(f'{OUTPUTPATH}cavity_{PROP}_{TIME_FILE_PATH}.csv', 'x') as content:
    content.write('NU,X,Y,Ux,Uy\n')
    for line in DATA_ARRAY:
        content.write(",".join([str(p) for p in line]))
        content.write('\n')

print("Finished Extraction!")
