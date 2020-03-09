"""
File: auxiliar_functions.py
Author: ShogunHirei
Description: Funções utilizadas repetidamente durante a implementação.
"""

# Função utilizada em ciclone_ANN para obter a estrutura da rede no output
import re
import os
import sys
import numpy as np
import tensorflow as tf
from joblib import dump, load
from pandas import read_csv, concat, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.layers import Dense, Input, concatenate, Masking
from keras.models import Sequential
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
import gc

class TrainingData:
    """
    File: training_data.py
    Author: ShogunHirei
    Description: Script para gerar dados de treinamento para facilitar
                 a scrita do código.
                 Lembrar que `pattern` precisa ser única para cada `datafile`
                 e se refere à um dado descrito no nome do arquivo .csv,
                 e precisa de um caractere a mais para ser eliminado
    """
    ORDER = []
    save_dir='./'
    scaler_folder='./'
    info_folder='./'
    pattern=r'\d+\.?\d*_'

    def __init__(self, data_folder, scaler=MinMaxScaler, FACTOR=1):
        self.data_folder = data_folder
        self.scaler = scaler
        self.N_SAMPLES = len(os.listdir(self.data_folder))
        self.factor = FACTOR
        # Para a geometria o factor é 5283.80102

    def data_gen(self, inp_labels=['Inlet_U', 'Points'],
                 out_labels=['U'], test_split=0.2, mag=['U'],
                 load_sc=True, seed=891859, **kargs):
        """
            File: training_data.py
            Function Name: data_gen
            Summary: Gerar dados de treinamento
            Description: Usar pasta com arquivos .csv e gerar conjuntos
                         de treinamento e teste.

            mag -> None ou Lista das magnitudes que SERÃO INSERIDAS no
                   conjunto de dados do treinamento
            load_sc -> Se for para carregar os scalers salvos
                         (verificação será feita na pasta `scaler_folder`)
        """
        mask =  kargs.get('mask', False)
        EVAL = kargs.get('EVAL', False)
        # NOTE: DETERMINAR MANEIRA DE FAZER PREVISÕES DE MULTIPLOS DADOS!!

        # Gerando {VEL_DE_ENTRADA : Dataframe} para todos os dados dentro da
        # pasta de dados `data_folder`
        print("Gerando dados de entrada...")
        # Problemas com a pattern para os arquivos gerados para redes recorrentes
        # arranjar uma maneira de adicionar todos os planos de valores de velocidade
        # para cada velocidade 
        _DF = {}
        for dado in os.scandir(self.data_folder):
            # TODO: Para outras patterns, verificar outras opções de
            #        slices e indexing
            # print("key", key)
            strg =f"Reading data: {dado.name + ' '*(30-len(dado.name))} |"
            strg +=f" Samples read: {len(_DF)}" 
            print(strg, end='\r', flush=True)
            key = float(re.findall(self.pattern, dado.name)[0][:-1])
            # Multiplicar a velocidade por uma constante para obter o número de Reynolds
            # do escoamento na entrada tangencial
            key = key * self.factor
            if key in _DF.keys():
                _DF[key].append(read_csv(dado.path))
            else:
                _DF[key] = [read_csv(dado.path)]
        print()
        # _DF = {float(): read_csv(dado.path)
               # for dado in os.scandir(self.data_folder)}

        # Verificando uniformidade dos dados e organizando em np.arrays
        # para facilitar manipulação de amostras
        print("Verificando labels")
        DATA = self.labels_read(_DF, MAG=mag)
        # Dados organizados com base em variáveis:
        #   DATA['Váriavel'] = [amostra1, amostras2, amostra3, ..., amostraN]

        # Para corrigir problemas com a diferença na magnitude dos dados
        #   preencher (padding) dados com valores np.nan (para evitar problemas
        #   na etapa de normalização) 
        ## Determinar o comprimento máximo da sequencia
        if mask:
            print("Determinando comprimento máximo de sequência")
            dummy = []
            for name in list(DATA.keys()):
                for sample in DATA[name]:
                    dummy.append(sample.shape)
            max_length = max(max(set(dummy)))
            print("Comprimento máximo: ", max_length)

            if len(set(dummy))>1:
                print("Padding values with np.nan...")
                for label in DATA.keys():
                    DATA[label] = pad_sequences(DATA[label], maxlen=max_length, padding='post', 
                                                value=np.nan, dtype='float32')
                    print(f"Padded {label}: Ok!")
            else:
                print("No padding necessary...")

        # Separar dados de input e output
        print("Separando dados de entrada e saída...")
        DATA = self.data_filter(DATA, inp_labels, out_labels)
        # Para que todos os labels sejam inseridas no dicionário 
        # para realizar a padronização
        # _TMP = DATA[0].copy()
        # _TMP.update(DATA[1])

        del _DF
        gc.collect()

        # Carregando normalizadores
        print("Carregando padronizadores...")
        scaler_dic = self.return_scaler(load_sc=load_sc, 
                                        data_input=DATA)

        # Remove repeated keys
        [DATA[0].pop(it, 'Erro!') for it in list(DATA[1].keys())]
        
        # Cada valor do dicionário scaler_dic referencia seu padronizador
        # utilizando isso para escalonar os dados
        print("Transformando dados...")
        for label in scaler_dic.keys():
            if label in DATA[0].keys():
                if bool(re.match("^Points_(0|1|2)", label)):
                    # DATA[0][label] representa os valores do dicionário
                    # Inserir zeros para acertar dimensões dos valores
                    # DATA[0][label] = np.array([scaler_dic[label].transform(DATA[0][label][n].reshape(-1, 1))
                                               # for n in range(len(DATA[0][label]))])
                    DATA[0][label] = scaler_dic[label].transform([DATA[0][label][n] 
                                                                  for n in range(len(DATA[0][label]))])
                    # print(label, DATA[0][label].shape)
                else:
                    DATA[0][label] = scaler_dic[label].transform(DATA[0][label])
                print(label, np.nanmin(DATA[0][label]), np.nanmax(DATA[0][label]))
            elif label in DATA[1].keys():
                DATA[1][label] = scaler_dic[label].transform(DATA[1][label])
                print(label, np.nanmin(DATA[1][label]), np.nanmax(DATA[1][label]))
                # print(label, DATA[1][label].max(), DATA[1][label].min())

        # Liberando espaço na memória
        # del _DF, _TMP

        # Para facilitar a consulta da ordem
        self.ORDER = []
        # Inputs
        self.ORDER.append({label: (indx, DATA[0][label].shape[1:])
                           for indx, label in enumerate(DATA[0].keys())})
        # Outputs
        self.ORDER.append({label: (indx, DATA[1][label].shape[1:])
                           for indx, label in enumerate(DATA[1].keys())})
        print("ORDER Ready!")
        
        # Separando os dados dos inputs e outputs da rede
        X = np.concatenate(tuple((data[..., np.newaxis]
                                  for data in DATA[0].values())), axis=2)
        Y = np.concatenate(tuple((data[..., np.newaxis]
                                  for data in DATA[1].values())), axis=2)

        del DATA
        gc.collect()
        
        if mask:
            print('Changing mask back to know value...')
            X[ np.isnan(X) ] = -101
            Y[ np.isnan(Y) ] = -101
        print(f'X: {X.min()} | {X.max()}')
        print(f'Y: {Y.min()} | {Y.max()}')
        print("Ok!")


        if EVAL:
            print("Full Data")
            print("Shape of X: ", X.shape)
            print("Shape of Y: ", Y.shape)
            return X, Y
        
        else:
            print("Gerando dicionário de treinamento...")
            (X_TRAIN, X_TEST, 
             Y_TRAIN, Y_TEST) = train_test_split(X, Y,
                                                 test_size=test_split, 
                                                 random_state=seed)

            print("Shape of X_TRAIN: ", X_TRAIN.shape)
            print("Shape of Y_TRAIN: ", Y_TRAIN.shape)
            print("Shape of X_TEST: ", X_TEST.shape)
            print("Shape of Y_TEST: ", Y_TEST.shape)
            del X,Y
            gc.collect()

            return (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)


    def training_dict(self, DICT, n):
        """
            File: auxiliar_functions.py
            Function Name: training_dict
            Summary: Organizar dados para 'fit'
            Description: Função para gerar dicionário com os nomes dos inputs
                         e outputs associados aos dados específicos
                         Utilizando o self.ORDER como referência
                         n ->  0 (input) ou 1 (output)
        """
        TRAINING_DICT = {}
        for label in self.ORDER[n].keys():
            # ORDER é uma lista com dois dicionários: Input e Output
            # Cada chave dos dicionários contem o índice dos dados nos
            # conjuntos de treinamento e teste e a dimensão
            TPL = [dim for dim in DICT.shape[:-1]]
            TPL += [1]
            TPL = tuple(TPL)
            TRAINING_DICT[label] = DICT[..., self.ORDER[n][label][0]].reshape(TPL)
        return TRAINING_DICT


    def mag_data_gen(self, labeled_data, pop_labels=['Points']):
        """
            File: training_data.py
            Function Name: mag_data_gen
            Summary: Gerar dados de magnitude da velocidade para concatenação
            Description: Função interna para agregar dados de magnitude
                         pop_labels -> colunas que terão a sua magnitude
                                       calculada
        """

        vector = {}
        if pop_labels:
            if isinstance(pop_labels, list):
                # Procura por todos os componentes em todas as labels para
                # identificar as labels seguidas por um número
                for pop_key in pop_labels:
                    for label in labeled_data.keys():
                        vec_str = label.split('_')
                        if bool(re.match(f'^{pop_key}', label)):
                            vector[vec_str[0]] = [comp for comp in labeled_data.keys()
                                                  if re.findall(f'^{vec_str[0]}_\d', comp)]
                # print(vector)
                # print(labeled_data.keys())
                for vec in vector.keys():
                    # try:
                    if isinstance(vector[vec], list):
                        # Lista de Dataframes que serão adicionados à labeled data
                        vec_mag = []
                        vec_mag += [concat([labeled_data[comp][n] for comp in vector[vec]],
                                            axis=1).apply(lambda x: np.sqrt(np.sum(x**2)),
                                                          axis=1) for n in len(labeled_data[vec])]

                    # vec_mag = np.concatenate(tuple([labeled_data[key][..., np.newaxis]
                                              # for key in vector[vec]]), axis=2)
                        labeled_data[f'{vec}_mag'] = vec_mag
                    # except:
                        # print("Não consegui calcular as magnitudes! ¯\_(ツ)_/¯")
            else:
                print('Apenas lista ou NoneType')

        return None

    def return_scaler(self, load_sc=True, data_input=None):
        """
            File: auxiliar_functions.py
            Function Name: return_scaler
            Summary: Retornar as funções utilizadas na padronização dos dados
            Description: Função que retorna os padronizadores para
                         serem reutilizados de outras formas além da geração
                         do conjunto de dados de treinamento e teste
        """
        # Considerando que `data_input` igual `labels` de self.labels_read
        SCALER_DICT = {}

        if load_sc:
            if data_input:
                for key in data_input:
                    for label in key:
                        SCALER_DICT[label] = load(f'{self.scaler_folder+label}.joblib')
            else:
                for fn in os.scandir(self.scaler_folder):
                    if bool(re.match(f'\w*.joblib$', fn.name)):
                        SCALER_DICT[fn.name.split('.joblib')[0]] = load(f'{fn.path}')
        else:
            print("Realizando normalização...")
            for key in data_input:
                for label in key:
                    if bool(re.match(f'^Points', label)):
                        # Cosiderando que a normalização será feita no plano 
                        # SCALER_DICT[label] = self.scaler().fit(data_input[label][..., np.newaxis][0])
                        dt = key[label][0][0].reshape(-1, 1)
                        SCALER_DICT[label] = self.scaler().fit(dt)
                        # CASO TENHAM AMOSTRAS DIFERENTES, ELIMINE O "if" E MANTENHA 
                        # APENAS A EXPRESSÃO DO ELSE ABAIXO
                    else:
                        SCALER_DICT[label] = self.scaler().fit(key[label])
                    # print(label, data_input[label].shape, 
                          # np.nanmin(data_input[label][np.where( data_input[label] != np.nan)]), 
                          # np.nanmax(data_input[label][np.where( data_input[label] != np.nan)]))
                    # print("=="*20)
                    dump(SCALER_DICT[label], f'{self.scaler_folder+label}.joblib')

        return SCALER_DICT

    
    def batch_prediction(self, model, INP_LIST, EVAL_ORDER):
        """
            File: auxiliar_functions.py
            Function Name: batch_prediction
            Summary: Generate data for the full grid of the ciclone
            Description: model -> keras model to predict data, 
                         INP_LIST -> list of results ready to insert in 
                                     model.predict
                         EVAL_ORDER, dict with inputs keys and ORDER
        """
        
        results = []
        dfs = []
        reescaled_dataframes = []
        # Considering that the INP_LIST correspond to `X = EVAL.training_dicy(...
        # Every dictionary has the keys of EVAL.ORDER 
        cols = list(EVAL_ORDER[0].keys()) + list(self.ORDER[1].keys())
        for item in INP_LIST:
            PRED = model.predict(item)
            temp = [item[p] for p in item.keys()] + PRED
            # Concatenated input data with output data to transform in pandas 
            # dataframe object
            temp = [item.reshape(-1, 1) for item in temp]
            results.append(temp)
        for it in results:
            # Concatenating cols axis 
            temp = np.concatenate(tuple(it), axis=1)
            # np.delete(temp,temp[temp == -101.0])
            # Removing masked values 
            temp = DataFrame(temp, columns=cols)
            # For the entries in the DataFrame that are not equal to the 
            # masking value
            temp = temp[temp != -101.0]
            dfs.append(temp)
        # As verified in previous tries, the NN will still predict values to 
        # masked indexes, so, it is needed one more postprocessing step
        ## First, return data to original scale (to this step make sense, 
        ##        the evaluation data HAS to be normalized with the same scalers
        scalers = self.return_scaler(load_sc=True)
        # labels = list(scalers.keys())
        for item in dfs:
            temp = []
            for label in cols:
                # Reshaping to (-1, 1) to later concatenation
                dm1 = item[label].to_numpy().reshape(1, -1)
                t1 = scalers[label].inverse_transform(dm1).reshape(-1, 1)
                temp.append(t1)
            # Concatenating along the columns axis
            temp = np.concatenate(tuple(temp), axis=1)
            temp = DataFrame(temp, columns=cols)
            # Here, the data is meant to be returned to original scale
            #       and should have `nan` objects in their rows
            # The second step is to remove the rows with `nan`
            dummy = temp[temp['Points_0'].isna()].copy()
            temp = temp.drop(dummy.index)
            reescaled_dataframes.append(temp)
        # Append all Dataframes along the row index
        final = concat(reescaled_dataframes, axis=0)
        print('Lenght of the result Dataframe:', len(final))

        return final


    def data_filter(self, data, inputs, outputs):
        """
            File: auxiliar_functions.py
            Function Name: data_filter
            Summary: Separar inputs e outputs da rede
            Description: Gerar tupla (X, Y) com as amostras de treinamento
                         e organizar para inserir em `train_test_split`
        """
        # Considerando que `data` é um dicionário com os dados
        # no qual as chaves são as propriedades e os valores é o conjunto de amostras
        X = {}
        Y = {}
        for label in data.keys():
            for entry in inputs:
                if re.findall("^"+entry, label):
                    X[label] = data[label]
            for entry in outputs:
                if re.findall("^"+entry, label):
                    Y[label] = data[label]
        print('Inputs:', " ".join(X.keys()))
        print('Outputs: ', " ".join(Y.keys()))
        # Dados não incluídos
        NI = set(data.keys()) - set(Y.keys()) - set(X.keys())
        del data
        if NI:
            print(", ".join(NI) +" not included!")
        data = (X, Y)
        return data

    def labels_read(self, sample_data, MAG=False, force_check=False):
        """
            File: auxiliar_functions.py
            Function Name: labels_read
            Summary: Ler todas as colunas
            Description: Função que lê todas as entradas e retorna os dados
                         organizados de acordo com os nomes das colunas.
        """
        labels = {}
        try:
        # Considerando que `sample_data` tenha a mesma estrutura que
        # `_DF`, ou seja, um dicionário com chaves {'Vel': [DF1, DF2, ..., DFn]}
        # Verificar primeiro se todos os dados possuem as mesmas colunas
            column_check = []
            if force_check==True:
                print("Force Check activated!! Be certain about the data folder!!")
                check = True
            else:
                gen = (open(FILE.path, 'r').readline() for FILE in os.scandir(self.data_folder))
                if len(set(gen)) == 1:
                    check = True
                else:
                    print('Data folder is not uniform')
            if check:
                print('Uniform data checked!!')
                # Usando todas as colunas dos dados para gerar dicionário
                # separados por colunas
                # Para garantir que os dados estarâo ordenados
                zipped_data = list(sample_data.items())
                # Pegando os nomes das labels como chaves para o dicionário
                cols = zipped_data[0][1][0].columns
                # Adicionar listas vazias para posterior transformação em array
                labels = dict.fromkeys(cols, [])
                # Atualizando chave de Velocidade de entrada
                labels['Inlet_U'] = []
                # Velocidade na entrada
                for data in zipped_data:
                    # para diminuir as iteraçõe sobres os dados
                    for line in data[1]:
                        # Adicionando a coluna com os dados da velocidade de entrada
                        Inlet_U = DataFrame(data[0] * np.ones((line.shape[-2], 1)), 
                                            columns=['Inlet_U'])
                        # Criando variaveis para manipulação dos DataFrames do Pandas
                        dummy1 = concat([line, Inlet_U ], axis=1)
                        cols = list(dummy1.columns)
                        for name in cols:
                            # Váriavel temporária 
                            temp = labels[name] + list((dummy1[name],))
                            labels[name] = temp
            else:
                print("Verifique pasta com dados!")
            if MAG:
                try:
                    self.mag_data_gen(labels, pop_labels=MAG)
                except:
                    print("Unexpected Error", sys.exc_info()[0])
                    print("Erro na determinação das magnitudes")
        except UnicodeDecodeError:
            print("""Cheque diretório com dados! Deve conter apenas arquivos .csv com
                    dados (colunas idênticas) para conjunto de treinamento""")
        except:
            print("Data is not uniform for separation!")

        for label in list(labels.keys()):
            corr_name = re.sub(r'[^A-Za-z0-9.][^A-Za-z0-9_.\-/]*', '_',
                              label)
            if corr_name != label:
                labels[corr_name] = labels[label]
                del labels[label]

        # Verificando valores dentro do último dicionário 
        # for key in labels.keys():
            # print(key, len(labels[key]), set([p.shape for p in labels[key]]))
            # print("="*30)

        return labels


    def list_callbacks(self, DIR, monit='val_loss'):
        """
            File: auxiliar_functions.py
            Function Name: list_callbacks
            Summary: Criar lista de callbacks comuns
            Description: Criar callback para Tensorboard entre outros
                         monit -> string para mudar variável observada para
                                  função loss
        """
        # Criando Callbacks para o treinamento
        CALLBACKS = [
                     # Tensorboard
                     TensorBoard(log_dir=DIR, histogram_freq=100, write_grads=False,
                                 write_images=False),
                     # Interromper Treinamento
                     EarlyStopping(monitor=monit, min_delta=0.00001, patience=175,
                                   restore_best_weights=True),
                     # Reduzir taxa de aprendizagem
                     ReduceLROnPlateau(monitor=monit, factor=0.1, patience=70, verbose=1,
                                       min_lr=1E-10)
                    ]
        return CALLBACKS


    
    def pickup_data(self, Val_Data, RND=None, key='Inlet_U'):
        """
            File: auxiliar_functions.py
            Function Name: pickup_data
            Summary: Preparar dados de input para previsão
            Description: Preparar dados que serão utilizados na previsão
                         Val_Data --> DICIONÁRIO com os dados de validação
                         RND --> Se None ou False usar valor aleatório
                                 se for especificado, será buscado o valor mais 
                                    próximo a ele no banco de dados
                         key --> chave que será usada para diferenciar entre os 
                                 dados
                         return suffix (str), caso_name (str), inp_data (dict)
        """
        scaler_inlet = self.return_scaler(load_sc=True)[key]
        array = Val_Data[key]
        # Retornando todos os valores de Inlet para a escala original 
        all_cases = [scaler_inlet.inverse_transform(array[n].reshape(1,-1))[...,0]
                     for n in range(len(array))]

        try:
            dist = [abs(RND-i[0]) for i in all_cases ]
            RND = dist.index(min(dist))
        except:
            RND = np.random.randint(0,len(all_cases))
            print("Apenas números, cara!")
        print("Caso comparado é ", all_cases[RND])
        suffix = str(all_cases[RND][0])

        # criando dicionário para entrar como dados
        inp_data = {e:Val_Data[e][RND][np.newaxis, ...] for e in Val_Data.keys()}

        print("Buscando dados originais do caso para comparação...")
        for dado in os.scandir(self.data_folder):
            # Verificação pelo arrendondamento dos valores padronizados
            #   comparados com a string do arrendondamento dos valores
            #   descritos no nome dos arquivos
            a = round(all_cases[RND][0], 3);
            b = round(float(re.findall(self.pattern, dado.name)[0][:-1]), 3)
            if bool(re.findall(f'^{str(a)}$', str(b))):
                caso_name = dado.name
        print("O caso para predição e comparação será:", caso_name)
        return suffix, caso_name, inp_data

     
    def predict_data_generator(self, model, INPUT_DATA, FILENAME, ORIGIN_DATA=None):
        """
            File: auxiliar_functions.py
            Function Name: predict_data_generator
            Summary: Gerar dados de previsão
            Description: Função que salva os dados de previsão e computa
                         a diferença entre o dado real e o dado previsto
                         INPUT_DATA -> Dicionário com os dados já escalados
                                       para inserir em model.predict
        """
                                
        print("Gerando dados para previsão")
        scaler_dict = self.return_scaler(load_sc=True)
        print("Carregado padronizadores!")

        # Dados inseridos na predição
        PREDICs = model.predict(INPUT_DATA)

        # Retornando os dados para a escala original do problema para
        # comparação com os dados de simulação

        # Nomes das colunas nos dados
        colunas = read_csv(os.scandir(self.data_folder).__next__().path).columns

        # Retornando os dados para a escala anterior
        in_col, out_col = [], []
        if isinstance(PREDICs, list):
            data_to_concat = []
            for name in INPUT_DATA.keys():
                # Substituir o "_" por uma regex para bater com os nomes das colunas
                col = [key for key in colunas if bool(re.match(f"^{name.replace('_', '[:_]')}$", key))]
                if bool(col):
                    in_col.append(col[0])
                    arr = scaler_dict[name].inverse_transform(INPUT_DATA[name].reshape(1, -1))
                    data_to_concat.append(DataFrame(arr.reshape(-1), columns=col))
            for name in self.ORDER[1].keys():
                col = [key for key in colunas if bool(re.match(f"^{name.replace('_', '[:_]')}$", key))]
                if bool(col):
                    out_col.append(col[0])
                    # ORDER[1] é o dicionário dos outputs que contém
                    # uma tupla com o indice e o shape do output
                    pred_arr = PREDICs[self.ORDER[1][name][0]].reshape(1,-1)
                    arr = scaler_dict[name].inverse_transform(pred_arr)
                    data_to_concat.append(DataFrame(arr.reshape(-1), columns=col))

        print(f"Inputs = {in_col}")
        print(f"Outputs = {out_col}")

        # Geração de arquivo .CSV para leitura
        SLICE_DATA = concat(data_to_concat, sort=True, axis=1)

        # Escrevendo o header no formato do paraview
        with open(self.save_dir+FILENAME, 'w') as filename:
            HEADER = ''
            for col in list(SLICE_DATA.columns):
                HEADER += '\"' + col + '\",'
            filename.write(HEADER[:-1])
            filename.write('\n')

        SLICE_DATA.to_csv(self.save_dir + FILENAME, index=False, header=False, mode='a')
        print("Dados de previsão copiados!")

        # Diferença do valor previsto e o caso original
        if ORIGIN_DATA:
            print("Calculando diferença...")
            ORIGIN_DF = read_csv(ORIGIN_DATA)

            DIFF = SLICE_DATA[out_col] - ORIGIN_DF[out_col]

            RESULT_DATA = concat([DIFF, SLICE_DATA[in_col]], axis=1)

            print('Escrevendo dados DIFERENÇA')
            name = re.findall(self.pattern, ORIGIN_DATA.split(self.data_folder)[-1])[0][:-1]
            RESULT_DATA.to_csv(self.save_dir + f'DIFF_SLICE_U_{name}.csv', index=False)
            print('Dados de diferença copiados!')
        return None

    
    def U_for_OpenFOAM(self, DATA, FILENAME):
        """
            File: auxiliar_functions.py
            Function Name: U_for_OpenFOAM
            Summary: Write data in format of OpenFoam 
            Description: Quick function for writing data in OF format
                         for substitution in velocity file
                         DATA (pandas.DataFrame): data to write
                         FILENAME (str): filename with path
        """

        with open(FILENAME, 'w') as fn:
            fn.write('(' + DATA.to_csv(None, index=False,  header=0, sep=' ', 
                                       line_terminator=')\n(')[:-1])
        
        return None


    def wall_data(self, XZ_DATA):
        """
            File: auxiliar_functions.py
            Function Name: wall_data
            Summary: Extrair dados do slice
            Description: Criar dados para a inserção na função zero_wall_mag
        """
        wall = []
        # Vetor A corresponde a diferença dos pontos entre o meio e extremo
        # para selecionar os pontos da parede
        # PNT1: (med1, med2) centro
        # PNT2: (max1, med2) extremo no eixo 1 e centro no eixo2
        A = np.array([np.mean(XZ_DATA[:, 0]) - np.max(XZ_DATA[:, 1]), 0])
        for pnt in XZ_DATA:
            # Vetor B, vetor do ponto analisado
            B = np.array(pnt) - np.array([np.mean(XZ_DATA[:, 0], axis=0),
                                          np.mean(XZ_DATA[:, 1])])
            B_mag = np.sqrt(np.sum(np.square(B)))
            A_mag = np.sqrt(np.sum(np.square(A)))
            if B_mag >= A_mag*0.99:  # 0.99 correção para pnts na parede
                tmp = [p for p in pnt]
                tmp.append(1)
                wall.append(np.array(tmp))
            else:
                wall.append(np.array([0, 0, 0]))
        wall = np.array(wall).reshape(-1, 3)
        return wall



class NeuralTopology:
    """
    Classe para reduzir código, automatizar criação de estrutura de rede
    Atributos:
        --> DISTRIBUITION = 'autoencoder'
        --> INICIAL_LAYER = 64
        --> ACTIVATION = 'tanh'
    """
    DISTRIBUTION = 'autoenconder'
    ACTIVATION = 'tanh'

    def __init__(self, MODEL=Sequential(), lyr_type=Dense, num_lyrs=5,
                 init_lyr=64):
        self.model = MODEL
        self.type = lyr_type
        self.num_lyrs = num_lyrs
        self.layer0 = init_lyr
        if type(self.model) == type(Sequential()):
            self.DISTRIBUTION = 'linear'

    
    def layer_stack_creation(self, deep, width, rate, **kargs):
        """
            File: auxiliar_functions.py
            Function Name: layer_stack_creation
            Summary: Generate layers of network
            Description: Generate layers with specified hyperparameters 
                             for the ditribution in self.DISTRIBUTION
                         deep (int) -> numbers of layers 
                         width (int) -> number of initial neurons to start 
                         rate [int, float]-> rate of change of numbers of neurons
                                             rate[0]^(-idx * rate[1])
        """
        if kargs:
            if kargs['kernel_initializer']:
                k_init = kargs['kernel_initializer']
            if kargs['activation']:
                activ = kargs['activation']
        else:
            k_init = 'he_normal'
            activ = self.ACTIVATION

        if self.DISTRIBUTION == 'autoenconder':
            LYR_STCK = [self.type(int(width*rate[0]**(-i*rate[1])),
                                  kernel_initializer=k_init,
                                  activation=activ) 
                        for i in range(deep) if i<=deep//2]
            LYR_STCK += [self.type(int(width*rate[0]**((i-deep)*rate[1])), 
                                   kernel_initializer=k_init,
                                   activation=activ) 
                         for i in range(deep) if i>=deep//2 ]
        else:
            LYR_STCK = [self.type(width,
                                  kernel_initializer=k_init,
                                  activation=activ) 
                        for i in range(deep) if i<=deep]

        return LYR_STCK

    def create_sequential(self, inputs=(1,), outputs=1):
        """
            File: auxiliar_functions.py
            Function Name: create_sequential
            Summary: Criar rede simples
            Description: Criar modelo Sequential simples de acordo com os
                         parametros da classe.
        """

        self.model.add(self.type(self.layer0, input_shape=inputs,
                                 activation=self.ACTIVATION))

        num_neurons = self.layer0
        # Adicionando rede do tipo autoencoder
        if self.DISTRIBUTION == 'autoencoder':
            for layer in range(self.num_lyrs):
                # Reduzindo numeros de neuronios até cerca da metade
                if layer <= self.num_lyrs/2:
                    num_neurons = num_neurons/2 if num_neurons % 2 == 0 else (num_neurons+1)/2
                    num_neurons = int(num_neurons)
                else:
                    num_neurons *= 2
                # Adicionando camadas na rede
                self.model.add(self.type(num_neurons, activation=self.ACTIVATION))

            # Adicionando camada final antes do output, igual a primeira
            self.model.add(self.type(self.layer0, activation=self.ACTIVATION))
        elif self.DISTRIBUTION == 'linear':
            # Adicionando camadas com mesmo número de neuronios
            for layer in range(self.num_lyrs):
                self.model.add(self.type(num_neurons, activation=self.ACTIVATION))
        # Camada do output
        self.model.add(self.type(outputs, activation=self.ACTIVATION))

        return self.model



    def multi_In_Out(self, INPUTS, OUTPUTS, LAYER_STACK=[], ADD_DENSE=True, MASKING=None):
        """
            File: auxiliar_functions.py
            Function Name:  multi_In_Out
            Summary: Criar camadas de input e output
            Description: APENAS PARA REDES TIPO MODEL, usar os 'dicionários'
                         INPUTS e OUTPUTS para gerar camadas com os nomes
                         descritos nas respectivas chaves.
                         --> Usar TrainingData.ORDER para obter os dicionários
                             de INPUTS e OUTPUTS
                         É compilado normalmente caso `self.model = Model()`
                         ###>>> GRAFOS ESTÃO DESCONECTADOS!!! (verificação)
        """
        # Para armazenamento das camadas
        INP_LAYERS = []
        TO_CONC = []
        OUT_LAYERS = []

        # Para inicializar LAYER_STACK
        if not LAYER_STACK:
            LAYER_STACK.append(self.type(self.layer0, activation=self.ACTIVATION))

        if type(self.model) != type(Sequential):
            print('Tipo de rede ok!')
            # Adicionando as camadas por nome
            for label in INPUTS.keys():

                ###### Usando função re.sub para ajustar nomes as regras de
                ######  nomes de variáveis do Tensorflow
                # https://www.tensorflow.org/api_docs/python/tf/Operation
                # Basicamente, remover caracteres especiais tipo: ':' e ')',
                # correc_label = re.sub(r'[^A-Za-z0-9.][^A-Za-z0-9_.\-/]*', '_',
                                      # label)
                # DEPRECATED! 
                correc_label = label
                in_lyr = Input(shape=(INPUTS[label][1][-1], 1), dtype='float32', name=correc_label)

                # Adicionando máscara para valores com dimensões diferentes
                if MASKING:
                    lyr = Masking(mask_value=MASKING)(in_lyr)
                    TO_CONC.append(lyr)
                # Se é necessário adicionar uma camada densa para corrigir as
                # dimensões do neurônio
                if ADD_DENSE:
                    lyr = self.type(self.layer0, activation=self.ACTIVATION)(in_lyr)
                    TO_CONC.append(lyr)
                INP_LAYERS.append(in_lyr)
            # Concatenando todas as entradas
            if len(INP_LAYERS)>1:
                # Caso exista apenas uma entrada
                if not TO_CONC:
                    JOINED_LYRS = concatenate(INP_LAYERS, axis=-1)
                else:
                    JOINED_LYRS = concatenate(TO_CONC, axis=-1)
            else:
                if bool(INP_LAYERS):
                    JOINED_LYRS = INP_LAYERS[0]
                else:
                    print("No Inputs! Are you nuts?")
            # TODO: Verificar como chamar um agrupamento de camadas aqui
            # ---> SUGEST: USAR LISTA COM AS CAMADAS E CHAMAR POR ELEMENTO
            X = LAYER_STACK[0](JOINED_LYRS)
            # LAYER_STACK DEVE SER UMA LISTA DE OBJETOS DE LAYERS
            if len(LAYER_STACK) > 1:
                for idx in range(1, len(LAYER_STACK)-1):
                    X = LAYER_STACK[idx](X)
                X = LAYER_STACK[-1](X)
            for label in OUTPUTS.keys():
                # correc_label = re.sub(r'[^A-Za-z0-9.][^A-Za-z0-9_.\-/]*', '_',
                                      # label)
                lyr = self.type(1, dtype='float32', name=label)(X)
                OUT_LAYERS.append(lyr)
        else:
            print('APENAS PARA REDES TIPO Model!')

        return INP_LAYERS, OUT_LAYERS


    def set_info(self, MODEL, filename):
        """
            File: auxiliar_functions.py
            Function Name: set_result
            Summary: Escrever topologia da rede
            Description: Usar a função recursiva para persistir a configuração
                         da rede.
        """
        print("Inserindo imagem do modelo...")
        plot_model(MODEL, to_file=filename+'.png', show_shapes=True)
        print("Gravando informações sobre arquitetura...")
        with open(filename+'.txt', 'w') as fn:
            rec_function(MODEL.get_config(), fn)
            MODEL.summary(print_fn=lambda x: fn.write(x + '\n'))
        print('Done!')
        return None


def rec_function(dic, logfile):
    """
    Author: ShogunHirei
    Description: Função para iterar recursivamente por dicionário de
                 configuração de rede (model.get_config()) e obter as
                 características principais da topologia
    """
    if type(dic) == dict:
        for p in dic.keys():
            if type(dic[p]) == str:
                string = ''
                if len(re.findall('name', p)) >= 1:
                    if 'units' in dic.keys() or 'activation' in dic.keys():
                        string += ' '.join([str(p),  str(dic[p]),
                                            str(dic['units']),
                                            str(dic['activation'])])
                        logfile.write(string+'\n')
            elif type(dic[p]) == dict:
                rec_function(dic[p], logfile)
            elif type(dic[p]) == list:
                for y in range(len(dic[p])):
                    rec_function(dic[p][y], logfile)
    elif type(dic) == list:
        for p in range(len(dic)):
            rec_function(dic[p], logfile)


# Função loss Customizada para magnitude
def mag_diff_loss(y_pred, y_true):
    """
        File: mag_isolated_prediction.py
        Function Name: mag_loss
        Summary: Função de custo para rede neural
        Description: Loss que adiciona a diferença da magnitude como penalidade
    """
    # Magnitude dos valores reais
    M_t = K.sqrt(K.sum(K.square(y_true), axis=-1))
    # Magnitude dos valores previstos
    M_p = K.sqrt(K.sum(K.square(y_pred), axis=-1))

    return K.mean(K.square(y_pred - y_true), axis=-1) + K.abs(M_p - M_t)


def zero_wall_mag(y_pred, y_true, wall_val):
    """
        File: mag_isolated_prediction.py
        Function Name: mag_loss
        Summary: Função de custo para rede neural
        Description: Mudando o valor de y_pred para a condição de velocidade
                     nula na parede, e adicionando a diferença entre
                     as magnitudes da rede e experimentais.

        new_mag = functools.partial(zero_wall_mag, wall_val=U_arr, xz_dict=XZ')
        U_arr --> Array com os dados de velocidade na parede
        XZ --> dados mapeados com os pontos cartesianos
                (considerado plano cilíndrico, uma amostra)
    """
    #    # Magnitude dos valores reais
    M_t = K.sqrt(K.sum(K.square(y_true), axis=-1))
    # Magnitude dos valores previstos
    M_p = K.sqrt(K.sum(K.square(y_pred), axis=-1))

    # Mudar pontos da parede em Booleano, para transformar em binários
    # inverter o ponto binários, para que os pontos na parede seja iguais a 0
    # e os outros pontos iguais a 1, para multiplicar pelo tensor de vel
    wall_val = tf.cast(wall_val, dtype=tf.bool)
    wall_val = tf.expand_dims((tf.cast(wall_val, dtype=tf.float32) - 1) * -1,
                              axis=0)  # shape (1, 862, 3)
    tmp_tens = y_pred * wall_val
    # Tecnicamente 'tmp_tens'
    print(tmp_tens.get_shape())
    PNTY = K.mean(K.square(y_pred - tmp_tens), axis=-1)

    return K.mean(K.square(y_pred - y_true), axis=-1) + K.abs(M_p - M_t) + PNTY


# Classe para escrever dados de maneira organizada
class Writer:
    '''
    Classe para escrever dados de maneira organizada
    '''
    import os

    def __init__(self, DATA_ARRAY, Case, Props, Dir):
        self.data = DATA_ARRAY
        self.dim = self.data.shape
        self.name = Case
        self.props = Props
        self.folder = Dir
        self.samples = self.dim[0]

    def mk_path(self, dim, name_pattern, folder_name='', ind=''):
        '''
        Criação de pastas de forma recursiva
        '''
        if len(dim[:-2]) == 1:
            path = folder_name + f"{name_pattern[0]}/"
            self.os.makedirs(path)
            for folder_num in range(dim[:-2][0]):
                index = [int(p) for p in str(ind+f"{folder_num}").split(',')]
                index = tuple(index)
                name = self.props + "_" + str(folder_num)
                self.record(index, path, name)
        else:
            folder_name += f"{name_pattern[0]}_"
            for num in range(dim[:-2][0]):
                folder_name += f"{num+1}/"
                ind += f"{num},"
                new_dim = dim[1:]
                new_names = name_pattern[1:]
                self.mk_path(new_dim, new_names, folder_name, ind)
                folder_name = folder_name[:-2]
                ind = ind[:-2]

    def record(self, ind, DIR, NAME):
        "Function to record the values in array"
        with open(f'{DIR}/{NAME}', 'w') as datafile:
            for dataline in self.data[ind]:
                datafile.write(str(tuple(dataline)))
        # Inserir local e indices do array
