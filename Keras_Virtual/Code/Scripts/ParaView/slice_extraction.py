# -*- coding: UTF-8 -*-
"""
File: slice_extraction.py
Author: ShogunHirei
Description: Script utilizado para definir um macro no ParaView
             para extração de dados de um slice.

"""
import os
import sys
from paraview.simple import OpenFOAMReader, GetActiveView, Slice, Show, SaveData
from paraview.simple import PointDatatoCellData, UpdatePipeline, PassArrays
from paraview import servermanager

# Definin arquivo para Leitura
FOAM_FILE = sys.argv[1]
# Definindo nome de arquivo de saída
VEL = sys.argv[2]

# Configuração da malha para inserção do slice
BODY = OpenFOAMReader(FileName=FOAM_FILE)
BODY.MeshRegions = ['internalMesh']
BODY.CellArrays = ['p', 'U']

Show()

# Fixar tempo para último disponível
CAM = GetActiveView()
CAM.ViewTime = BODY.TimestepValues[-1]

# Implementação e configuração do slice
SL = Slice()
SL.Input = BODY
SL.SliceType = 'Plane'
SL.SliceType.Normal = [0, 1, 0]
SL.SliceType.Origin = [0, -0.2, 0]

Show()

# Passing PointDatatoCellData Filter
PDCD = PointDatatoCellData()
PDCD.Input = SL
PDCD.PassPointData = True

Show()
print(PDCD.PointData.NumberOfArrays)

# Passando para filtro para Eliminar tensor de Reynolds
PA = PassArrays(PDCD)
PA.PointDataArrays = ['U', 'p']

print(PA.PointData.NumberOfArrays)

# Extração de dados para persistência
FOAM_RUN = os.getenv('FOAM_RUN')
PATH = FOAM_RUN + '/../Ciclone/ANN_DATA/'
FILENAME = 'SLICE_DATA_U_'+VEL + '_.csv'
SaveData(PATH+FILENAME, Precision=5, proxy=PA, FieldAssociation='Points')
