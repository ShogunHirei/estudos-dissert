"""
File: full_grid_extraction.py
Author: ShogunHirei
Description: Script to save internal mesh data as CSV file
"""

import sys
from paraview.simple import CellCenters, Clip, SaveData, OpenFOAMReader
from paraview.simple import GetActiveView, FindViewOrCreate, PassArrays, Show

# FoamFILE for reading data
FOAM_FILE = sys.argv[1]

# Velocidade
VEL = sys.argv[2]

# Save Folder
SAVE_DIR = sys.argv[3]

# Reading the data of simulation
BODY = OpenFOAMReader(FileName=FOAM_FILE)
BODY.MeshRegions = ['internalMesh']
BODY.PointArrays = ['p', 'U', 'div(phi)', 'Res']
BODY.CellArrays = ['p', 'U', 'div(phi)', 'Res']

print BODY.TimestepValues[-1]
BODY.UpdatePipeline()

CC = CellCenters()
CC.Input = BODY
CC.VertexCells = 1

PA = PassArrays(Input=CC)
PA.PointDataArrays = ['p', 'div(phi)', 'U', 'Res']
PA.CellDataArrays = ['p', 'div(phi)', 'U', 'Res']
PA.UpdatePipeline()
print(PA.PointDataArrays)

# Saving point data information 
SaveData(SAVE_DIR + 'centroid_U_' + VEL + '_.csv', proxy=PA, Precision=8,
         FieldAssociation='Points')
