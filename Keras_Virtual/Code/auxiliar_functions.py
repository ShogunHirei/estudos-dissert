"""
File: auxiliar_functions.py
Author: ShogunHirei
Description: Funções utilizadas repetidamente durante a implementação.
"""


def write_output_vector(data,  NAME_PATTERN=['Amostra', ''], location='./'):
    """
        File: auxiliar_functions.py
        Function Name: write_output_vector
        Summary: Gerar arquivos com dados
        Description: Grava dados a partir de dados gerados localmente.
                        data -> dados que serão gravados, a amostra )
                        NAME_PATTERN -> os nomes dos arquivos gerados, 
                        Location -> Local para a gravação.
    """
    # Número de dados (amostras)
    N = data.shape[0]
    for amostra in data:
        FILENAME += NAME_PATTERN[1]
        with open(FILENAME) as datafile:
            datafile.write()


class Writer:
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
            for folder_num in range(dim[:-2][0]):
                path = folder_name + f"{name_pattern[0]}_{folder_num+1}"
            os.mkdir(path)
            index = [int(p) for p in str(ind+f"{folder_num}").split(',')]
            index = tuple(index)
            self.record(self.data[index], path)
        else:
            folder_name += f"{name_pattern[0]}_"
            for num in range(dim[:-2][0]):
                folder_name += f"{num+1}/"
                ind += f"{num},"
                new_dim = dim[1:]
                new_names = name_pattern[1:]
                mk_path(new_dim, new_names, folder_name, ind)
                folder_name = folder_name[:-2]
                ind = ind[:-2]

    def record(self, Dir, ind):
        with open(f'{DIR}/{PROP}', 'rw') as datafile:
           for  dataline in self.data[ind]:
               datafile.write(dataline) 
        # Inserir local e indices do array
    

