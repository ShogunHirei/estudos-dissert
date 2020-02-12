"""
File: extract_data_RNN.py
Author: ShogunHirei
Description: Script do paraview para extrair dados dos planos em sequencia para 
             testar o desempenho de redes recorrentes na previsao do perfil de 
             velocidade
"""

import sys
# from paraview.simple import CellCenters, Clip, SaveData, OpenFOAMReader
# from paraview.simple import GetActiveView, FindViewOrCreate, PassArrays, Show
from paraview.simple import *

# Script e executado atraves de linha de comando, 
# Definir como entrada  o arquivo openFOAM e o diretorio no 
# qual sera salvo os dados 

VEL = str(10)

FOAM_FILE = sys.argv[1]

SAVE_DIR = sys.argv[2]


# Abrindo a simulacao pelo Paraview
print("Reading objects...")
BODY = OpenFOAMReader(FileName=FOAM_FILE)
BODY.MeshRegions = ['internalMesh']
BODY.PointArrays = ['p', 'U', 'div(phi)', 'Res']
BODY.CellArrays = ['p', 'U', 'div(phi)', 'Res']
BODY.UpdatePipeline()


print("Update Camera")
Show()
print("Here!")
renderview = FindViewOrCreate('RenderView1', viewtype='RenderView')
print("Here2")
renderview.Update()
print("Here2")

print "Mudando tempo", "\r"
# Fixar tempo para ultimo disponivel
CAM = GetActiveView()
CAM.ViewTime = BODY.TimestepValues[-1]

renderview.Update()

# Obtendo os valores do centroides 
CC = CellCenters()
CC.Input = BODY
CC.VertexCells = 1

renderview.Update()

# Definindo os parametros iniciais 
Y_NORMAL = [0, 1, 0]
# Valor de corte para o primeiro clip
YMIN = -1.143
# Variacao entre os planos na direcao y
DY = 0.019

print("Primeiro ClipObject...", '\r')
# ClipMin
CP1 = Clip(Input=CC)
CP1.ClipType = 'Plane'
CP1.ClipType.Normal = Y_NORMAL
CP1.ClipType.Origin = [0, YMIN, 0]
CP1.Invert = 1

PA = PassArrays(Input=CP1)
PA.PointDataArrays = ['p', 'U', 'div(phi)', 'Res']
PA.CellDataArrays = ['p', 'div(phi)', 'U', 'Res']
PA.UpdatePipeline()

renderview.Update()

SaveData(SAVE_DIR + 'centroid_Y_' + str(YMIN) + '_.csv', proxy=PA, Precision=8,
         FieldAssociation='Points')

del PA
# Comecar laco para extrair os 54 planos de dados
YN = YMIN
print("Inicio do laco...")
for i in range(54):
    YN += DY
    CLIPN = Clip(Input=CC)
    CLIPN.ClipType = 'Plane'
    CLIPN.Normal = Y_NORMAL
    CLIPN.Origin = [0, YN, 0]
    CLIPN.Invert = 1

    renderview.Update()
    Show()

    CLIP_TO_SAVE = Clip(Input=CLIPN)
    CLIP_TO_SAVE.ClipType = 'Plane'
    CLIP_TO_SAVE.ClipType.Normal = Y_NORMAL
    CLIP_TO_SAVE.ClipType.Origin = [0, YN-DY, 0]
    CLIP_TO_SAVE.Invert = 0

    # Saving process
    CLIP_TO_SAVE.UpdatePipeline()
    renderview.Update()

    PA = PassArrays(Input=CLIP_TO_SAVE)
    PA.PointDataArrays = ['p', 'U', 'div(phi)', 'Res']
    PA.CellDataArrays = ['p', 'div(phi)', 'U', 'Res']
    PA.UpdatePipeline()

    renderview.Update()

    SaveData(SAVE_DIR + 'centroid_Y_' + str(YN) + '_.csv', proxy=PA, Precision=8,
         FieldAssociation='Points')

    print "Concluido passo ", i, '\r'

    del PA, CLIP_TO_SAVE, CLIPN



