#!-*- coding: utf-8 -*-
"""
File: cell_centers_clip.py
Author: ShogunHirei
Description: Script para extrair os centros das células próximos a y=-0.2
"""

import sys
from paraview.simple import CellCenters, Clip, SaveData, OpenFOAMReader
from paraview.simple import GetActiveView, FindViewOrCreate, PassArrays, Show

# Considerando que esse script será utilizado na linha de comando através do
# pvpython, os argumentos serão o nome do arquivo .OpenFoam e a velocidade de
# entrada
FOAM_FILE = sys.argv[1]

# Velocidade
VEL = sys.argv[2]

# Save Folder
SAVE_DIR = sys.argv[3]

# Abrindo a simulação pelo Paraview
BODY = OpenFOAMReader(FileName=FOAM_FILE)
BODY.MeshRegions = ['internalMesh']
BODY.PointArrays = ['p', 'U', 'div(phi)', 'Res']
BODY.CellArrays = ['p', 'U', 'div(phi)', 'Res']
BODY.UpdatePipeline()

Show()
renderview = FindViewOrCreate('RenderView1', viewtype='RenderView')
renderview.Update()

# Fixar tempo para último disponível
CAM = GetActiveView()
CAM.ViewTime = BODY.TimestepValues[-1]

renderview.Update()

# Centralizando as células
CC = CellCenters()
CC.Input = BODY
CC.VertexCells = 1

renderview.Update()

# Clip1
CP1 = Clip(Input=CC)
CP1.ClipType = 'Plane'
CP1.ClipType.Normal = [0, 1, 0]
CP1.ClipType.Origin = [0, -0.22, 0]
CP1.Invert = 0
# CP1.Scalars = ['POINTS', 'div(phi)']
# CP1.Value = -0.002842850983142853

renderview.Update()

# Clip2
CP2 = Clip(Input=CP1)
CP2.ClipType.Normal = [0, 1, 0]
CP2.ClipType.Origin = [0, -0.19, 0]
CP2.Invert = 1
CP2.UpdatePipeline()
renderview.Update()

PA = PassArrays(Input=CP2)
PA.PointDataArrays = ['p', 'U', 'div(phi)', 'Res']
PA.CellDataArrays = ['p', 'div(phi)', 'U', 'Res']
PA.UpdatePipeline()

renderview.Update()

SaveData(SAVE_DIR + 'centroid_U_' + VEL + '_.csv', proxy=PA, Precision=8,
         FieldAssociation='Points')
