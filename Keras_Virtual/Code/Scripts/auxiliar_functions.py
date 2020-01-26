"""
File: auxiliar_functions.py
Author: ShogunHirei
Description: Funções utilizadas repetidamente durante a implementação.
"""

# Função utilizada em ciclone_ANN para obter a estrutura da rede no output
import re
import os
import numpy as np
from joblib import dump, load
from pandas import read_csv, concat, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.layers import Dense, Input, concatenate
from keras.models import Sequential
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


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
    pattern=r'\d+\.?\d*_'

    def __init__(self, data_folder, scaler=MinMaxScaler):
        self.data_folder = data_folder
        self.scaler = scaler
        self.N_SAMPLES = len(os.listdir(self.data_folder))

    def data_gen(self, inp_labels=['Inlet_U', 'Points:0', 'Points:2'],
                 out_labels=['U'], test_split=0.2, mag=['Points'],
                 load_sc=True, save_sc=False):
        """
            File: training_data.py
            Function Name: data_gen
            Summary: Gerar dados de treinamento
            Description: Usar pasta com arquivos .csv e gerar conjuntos
                         de treinamento e teste.

            mag -> None ou Lista das magnitudes que NÃO SERÃO INSERIDAS no 
                   conjunto de dados do treinamento
            load_sc -> Se for para carregar os scalers salvos
            save_sc -> não carregar, criar novos scaler e salvá-los
        """

        # Gerando {VEL_DE_ENTRADA : Dataframe} para todos os dados dentro da
        # pasta de dados `data_folder`
        _DF = {float(re.findall(self.pattern,
                                dado.path)[0][:-1]): read_csv(dado.path)
               for dado in os.scandir(self.data_folder)}

        # Verificando uniformidade dos dados e organizando em np.arrays
        # para facilitar manipulação de amostras
        DATA = self.labels_read(_DF, MAG=mag)

        DATA = self.data_filter(DATA, inp_labels, out_labels)
        _TMP = DATA[0].copy()
        _TMP.update(DATA[1])

        # Carregando normalizadores
        scaler_dic = self.return_scaler(load_sc=load_sc, save_sc=save_sc,
                                        data_input=_TMP)

        # Cada valor do dicionário scaler_dic referencia seu padronizador
        # utilizando isso para escalonar os dados
        for label in scaler_dic.keys():
            if label in DATA[0].keys():
                DATA[0][label] = scaler_dic[label].transform(DATA[0][label])
            elif label in DATA[1].keys():
                DATA[1][label] = scaler_dic[label].transform(DATA[1][label])

        # Liberando espaço na memória
        del _DF, _TMP

        # Para facilitar a consulta da ordem
        self.ORDER = []
        # Inputs
        self.ORDER.append({re.sub(r'[^A-Za-z0-9.][^A-Za-z0-9_.\-/]*', '_', 
                                      label): (indx, DATA[0][label].shape[1:])
                           for indx, label in enumerate(DATA[0].keys())})
        # Outputs
        self.ORDER.append({re.sub(r'[^A-Za-z0-9.][^A-Za-z0-9_.\-/]*', '_', 
                                      label): (indx, DATA[1][label].shape[1:])
                           for indx, label in enumerate(DATA[1].keys())})
        print("ORDER Ready!")

        # Separando os dados dos inputs e outputs da rede
        X = np.concatenate(tuple([data[..., np.newaxis]
                                  for data in DATA[0].values()]), axis=2)
        Y = np.concatenate(tuple([data[..., np.newaxis]
                                  for data in DATA[1].values()]), axis=2)

        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y,
                                                            test_size=test_split)

        print("Shape of X_TRAIN: ", X_TRAIN.shape)
        print("Shape of Y_TRAIN: ", Y_TRAIN.shape)
        print("Shape of X_TEST: ", X_TEST.shape)
        print("Shape of Y_TEST: ", Y_TEST.shape)

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


    def mag_data_gen(self, labeled_data, pop_labels):
        """
            File: training_data.py
            Function Name: mag_data_gen
            Summary: Gerar dados de magnitude da velocidade para concatenação
            Description: Função interna para agregar dados de magnitude
                         pop_labels -> colunas que NÃO SERÃO DETERMINADOS dados
                                       de magnitude
        """

        vector = {}
        if pop_labels:
            if isinstance(pop_labels, list):
                for label in labeled_data.keys():
                    # Identificando as componentes dos vetores para os nomes das colunas
                    vec_str = label.split(':')
                    # Se houver as componentes adicionar ao dicionário
                    # usando o ":" pq é o padrão do OpenFoam
                    if len(vec_str) > 1:
                        # verifica se a label está dentro do padrão de cada umas das
                        # strings listadas em pop_labels, e se estiver não adiciona
                        # ao dicionário `vector` que terão suas magnitudes inseridas
                        # no conjunto geral de dados
                        if not all([bool(re.findall(f'^{pop_key}', vec_str[0]))
                                 for pop_key in pop_labels]):
                            vector[vec_str[0]] = [comp for comp in labeled_data.keys() 
                                                  if re.findall(f'^{vec_str[0]}:\d', comp)]
                for vec in vector.keys():
                    # usando a lista gerada dos componentes ['X:1', 'X:2', ..., 'X:N']
                    # nas chaves dos dicionários e concatatenando os componentes dos 
                    # respectivos arrays
                    vec_mag = np.concatenate(tuple([labeled_data[key][..., np.newaxis]
                                              for key in vector[vec]]), axis=2)
                    labeled_data[f'{vec}_mag'] = np.sqrt(np.sum(vec_mag**2, axis=2))
            else:
                print('APENAS LISTA OU NONE')

        return None

    def return_scaler(self, load_sc=True, save_sc=False, data_input=None):
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
            for label in data_input.keys():
                SCALER_DICT[label] = load(f'{self.scaler_folder+label}.joblib')
        else:
            for label in data_input.keys():
                SCALER_DICT[label] = self.scaler().fit(data_input[label])
            # XZ = data_input[0]
            # INPUT_U = data_input[1]
            # U_xyz = data_input[2]
            # XZ_scaler = self.scaler().fit(XZ[0])
            # INPUT_U_scaler = self.scaler().fit(INPUT_U[:, 0])
            # Ux_scaler = self.scaler().fit(U_xyz[..., 0])
            # Uy_scaler = self.scaler().fit(U_xyz[..., 1])
            # Uz_scaler = self.scaler().fit(U_xyz[..., 2])
        if save_sc:
            for label in data_input.keys():
                dump(SCALER_DICT[label], f'{self.scaler_folder+label}.joblib')
            # dump(XZ_scaler, self.scaler_folder+'points_scaler.joblib')
            # dump(INPUT_U_scaler, self.scaler_folder+'U_input_scaler.joblib')
            # dump(Ux_scaler, self.scaler_folder+'Ux_scaler.joblib')
            # dump(Uy_scaler, self.scaler_folder+'Uy_scaler.joblib')
            # dump(Uz_scaler, self.scaler_folder+'Uz_scaler.joblib')

        # SCALER_DICT = {'XZ': XZ_scaler, 'U_in': INPUT_U_scaler,
                       # 'Ux_scaler': Ux_scaler, 'Uy_scaler': Uy_scaler,
                       # 'Uz_scaler': Uz_scaler}

        return SCALER_DICT


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
        NI = set(data.keys()) - set(Y.keys()) - set(X.keys())
        if NI:
            print(", ".join(NI) +" not included!")
        data = (X, Y)
        return data

    def labels_read(self, sample_data, MAG=False):
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
            # `_DF`, ou seja, um dicionário com chaves {'Vel': DataFrame}
            # Verificar primeiro se todos os dados possuem as mesmas colunas
            check = all([(sample_data[i].columns == sample_data[j].columns).all()
                         for i in sample_data.keys() for j in sample_data.keys()])
            if check:
                # Usando todas as colunas dos dados para gerar dicionário
                # separados por colunas
                Inlet_U = []
                # Para garantir que os dados estarâo ordenados
                zipped_data = list(sample_data.items())
                # Velocidade na entrada
                for VEL in [key[0] for key in zipped_data]:
                    # Todas as chaves são as velocidades de entrada
                    Inlet_U.append([VEL]*sample_data[VEL].shape[-2])
                labels['Inlet_U'] = np.array(Inlet_U)

                # As colunas dos dados (propriedades) dentro das amostras
                for label in zipped_data[0][1].columns:
                    # Selecionando os dados de acordo com os DataFrames
                    # organizados em `zipped_data`
                    arr = np.array([sample[label] for sample in [data[1]
                                                    for data in zipped_data]])
                    labels[label] = arr
                if MAG:
                    self.mag_data_gen(labels, pop_labels=MAG)
                    print(labels.keys())

                # TODO: Verificar possível utiilzação do Dataframe para essa organização
        except UnicodeDecodeError:
            print("""Cheque pasta com dados! Deve conter apenas arquivos .csv com 
                    dados (colunas idênticas) para conjunto de treinamento""")
        except:
            print("Data is not uniform for separation!")
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
        
    ## DEPRECATED!!
    def append_div_data(self, data, scaled_data):
        """
            File: auxiliar_functions.py
            Function Name: append_div_data
            Summary: Append Div data
            Description: Função que verifica se ´div(phi)´ está entre os
                         dados disponíveis e os adiciona ao conjunto de
                         treinamento
        """
        # Considerando uniformidade entre os dados
        # inserindo DF como a fonte de dados repetir o executado em data_gen
        try:
            DIV = np.array([np.array(sample[0]['div(phi)'])
                            for sample in data if 'div(phi)' in sample[0].columns])
            # Normalizando os dados para retornar o conjunto de dados
            # prontos para o treinamento
            Scaler_div = self.scaler().fit(DIV)
            scl_DIV = Scaler_div.transform(DIV)
            scaled_data = np.concatenate((scaled_data, scl_DIV[..., np.newaxis]), axis=2)
        except:
            print('No DIV inserted')
        return scaled_data

    def predict_data_generator(self, model, INPUT_DATA, FILENAME):
        """
            File: auxiliar_functions.py
            Function Name: predict_data_generator
            Summary: Gerar dados de previsão
            Description: Função que salva os dados de previsão e computa
                         a diferença entre o dado real e o dado previsto
        """
        print("Gerando dados para previsão")
        scaler_dict = self.return_scaler(load_sc=True)

        VEL_ARR = np.array([[10.0]*868]).reshape(-1, 1)
        VEL_ARR = scaler_dict['U_in'].transform(VEL_ARR).reshape(1, -1, 1)

        # Valores previstos para Ux, Uy e Uz
        PREDICs = model.predict(INPUT_DATA)
        print([p.shape for p in PREDICs])

        # Retornando os dados para a escala anterior
        Ux = DataFrame(scaler_dict['Ux_scaler'].inverse_transform(PREDICs[..., 0]).reshape(-1), columns=['U:0'])
        Uy = DataFrame(scaler_dict['Uy_scaler'].inverse_transform(PREDICs[..., 1]).reshape(-1), columns=['U:1'])
        Uz = DataFrame(scaler_dict['Uz_scaler'].inverse_transform(PREDICs[..., 2]).reshape(-1), columns=['U:2'])

        # Inserindo valor dos pontos de Y
        XYZ = read_csv(os.scandir(self.data_folder).__next__().path)[['Points:0', 'Points:1', 'Points:2']]

        # Geração de arquivo .CSV para leitura
        SLICE_DATA = concat([Ux, Uy, Uz, XYZ], sort=True, axis=1)

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
        print("Calculando diferença...")
        ORIGIN_DATA = read_csv(self.data_folder+'SLICE_DATA_U_10_0.csv')

        DIFF = SLICE_DATA[['U:0', 'U:1', 'U:2']] - ORIGIN_DATA[['U:0', 'U:1', 'U:2']]

        RESULT_DATA = concat([DIFF, XYZ], axis=1)

        print('Escrevendo dados DIFERENÇA')
        RESULT_DATA.to_csv(self.save_dir + 'DIFF_SLICE_U_10.csv', index=False)
        print('Dados de diferença copiados!')


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

    def organized_data(self, data):
        """
            File: auxiliar_functions.py
            Function Name: organized_data
            Summary: Extrair dados de forma organizada
            Description: Usa a magnitude de um vetor da origem à extremidade
                         e guarda os dados que estão em mesma condição (círculo)
                         manda para uma lista
        """
        # TODO: FALTA CONSERTAR MAGNITUDE PARA QUADRADO NO MEIO
        # NOTE: Adicionar etapa de verificação de dados, 'np.intersect1d'
        xz = np.concatenate((data, np.zeros((len(data), 1))), axis=1)
        # função de magnitude 'atalho'
        mag = lambda x: np.sqrt(np.sum(np.square(x)))

        # Vetor A de referência
        A = np.array([np.max(xz[:, 0]), np.mean(xz[:, 1]), 0]) - np.mean(xz, axis=0)
        arm2 = []
        MEAN_ORIG = np.mean(xz, axis=0)
        while mag(A) > 0.1455:
            arm1 = []
            for pnt in xz:
                # vetor B de comparação com A
                B = pnt - MEAN_ORIG
                if mag(B) >= 0.97 * mag(A):
                    # Adicionado um item a mais por causa do vetor velocidade
                    arm1.append([p for p in pnt])
                else:
                    # para manter as mesmas dimensões do mapeamento dos pontos '(868, 3)'
                    arm1.append([0, 0, 0])
            arm1 = np.array([np.array(p) for p in arm1])
            # Interseção dos dados no extremo e dados XZ, indices
            inter = np.intersect1d(xz[:, 0], arm1[:,0], return_indices=True)[1]
            # Para manter a dimensão dos arrays, mascarar dados com 0
            mask = np.ones(xz.shape, dtype=bool)
            # Mudando valores que identificados para 0
            mask[list(inter)] = 0
            NEW_XZ = xz * mask

            xz = NEW_XZ
            A = np.array([MEAN_ORIG[0], np.max(xz[:, 1:]), 0]) - MEAN_ORIG
            arm2.append(arm1)


    def data_prediction(self, MODEL, INPUT_DATA, FILENAME, NAME='NEW_SLICE_10_Isolated'):
        """
            File: auxiliar_functions.py
            Function Name: data_prediction
            Summary: Gravar dados preditos
            Description: Inserir modelo da rede e dados adicionais de entrada
                         necessários do modelo, como dicionário para a
                         predição.
        """
        scalers = self.return_scaler(load_sc=True)

        # TODO: Verificar a dimensão dos dados de predição
        #       Devido à futura inserção de novas variáveis
        PREDICs = MODEL.predict(INPUT_DATA)
        print([p.shape for p in PREDICs])

        # Para armazenar dados que serão transformados para retornar
        # às escalas anteriores
        DATA_DICT = {}

        OUTPUT_NAMES = MODEL.output_names
        # Retornando os dados para a escala anterior
        for name in OUTPUT_NAMES:
            if name in scalers.keys():
                DATA_DICT[name] = scalers[name].inverse_transform()

        # Transformando em DataFrames do panda para facilitar manipulação
        U_xyz = [DataFrame(dt, columns=i) for i, dt in DATA_DICT.items()]

        # Dado Original de Velocidade
        ORIGIN_DATA = read_csv(self.data_folder+FILENAME)

        # Coordenadas
        XYZ = ORIGIN_DATA[['Points:0', 'Points:1', 'Points:2']]

        # Geração de arquivo .CSV para leitura
        NEW_FILE = f'{NAME}.csv'

        SLICE_DATA = concat(U_xyz + XYZ, sort=True, axis=1)

        # Escrevendo o header no formato do paraview
        with open(self.save_dir+NEW_FILE, 'w') as filename:
            HEADER = ''
            for col in list(SLICE_DATA.columns):
                HEADER += '\"' + col + '\",'
            filename.write(HEADER[:-1])
            filename.write('\n')

        SLICE_DATA.to_csv(self.save_dir + NEW_FILE, index=False, header=False, mode='a')
        print("Dados de previsão copiados!")

        # Diferença do valor previsto e o caso original
        print("Calculando diferença...")

        DIFF = SLICE_DATA[['U:0', 'U:1', 'U:2']] - ORIGIN_DATA[['U:0', 'U:1', 'U:2']]

        RESULT_DATA = concat([DIFF, XYZ], axis=1)

        print('Escrevendo dados DIFERENÇA')
        RESULT_DATA.to_csv(self.save_dir + 'DIFF_SLICE_U_10.csv', index=False)
        print('Dados de diferença copiados!')

        return None


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



    def multi_In_Out(self, INPUTS, OUTPUTS, LAYER_STACK=[], ADD_DENSE=True):
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
                correc_label = re.sub(r'[^A-Za-z0-9.][^A-Za-z0-9_.\-/]*', '_', 
                                      label)
                in_lyr = Input(shape=(INPUTS[label][1][-1], 1), dtype='float32', name=correc_label)
                # Se é necessário adicionar uma camada densa para corrigir as
                # dimensões do neurônio
                if ADD_DENSE:
                    lyr = self.type(self.layer0, activation=self.ACTIVATION)(in_lyr)
                    TO_CONC.append(lyr)
                INP_LAYERS.append(in_lyr)
            # Concatenando todas as entradas
            if not TO_CONC:
                JOINED_LYRS = concatenate(INP_LAYERS, axis=-1)
            else:
                JOINED_LYRS = concatenate(TO_CONC, axis=-1)
            # TODO: Verificar como chamar um agrupamento de camadas aqui
            # ---> SUGEST: USAR LISTA COM AS CAMADAS E CHAMAR POR ELEMENTO
            X = LAYER_STACK[0](JOINED_LYRS)
            # LAYER_STACK DEVE SER UMA LISTA DE OBJETOS DE LAYERS
            if len(LAYER_STACK) > 1:
                for idx in range(1, len(LAYER_STACK)-1):
                    X = LAYER_STACK[idx](X)
                X = LAYER_STACK[-1](X)
            for label in OUTPUTS.keys():
                correc_label = re.sub(r'[^A-Za-z0-9.][^A-Za-z0-9_.\-/]*', '_', 
                                      label)
                lyr = self.type(1, dtype='float32', name=correc_label)(X)
                OUT_LAYERS.append(lyr)
        else:
            print('APENAS PARA REDES TIPO Model!')

        return INP_LAYERS, OUT_LAYERS


    def set_result(self, MODEL, filename):
        """
            File: auxiliar_functions.py
            Function Name: set_result
            Summary: Escrever topologia da rede
            Description: Usar a função recursiva para persistir a configuração
                         da rede.
        """
        with open(filename, 'w') as fn:
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
