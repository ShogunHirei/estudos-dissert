"""
File: pressDroppSCript.py
Author: ShogunHirei
Description: Script to calculate press drop at entrance
"""

import sys
from re import findall
from paraview.simple import PVFoamReader, Slice, IntegrateVariables

FOAMFILE = sys.argv[1]

# Read foam case
BODY = PVFoamReader(FileName=FOAMFILE)
BODY.VolumeFields = ['p', 'ANN_p']

SL = Slice(Input=BODY)
SL.SliceType = 'Plane'
SL.SliceType.Normal = [0, 0, 1]
SL.SliceType.Origin = [0, 0, 0.255]
SL.UpdatePipeline()

# Integrate Variables to get FieldAverage
IV = IntegrateVariables(Input=SL)
IV.DivideCellDataByVolume = 1
IV.UpdatePipeline()

# Show Cell Data, to get results
PINN_pres = IV.CellData.GetArray('ANN_p').GetRange()[0]
SIM_pres = IV.CellData.GetArray('p').GetRange()[0]
NAME = FOAMFILE.split()[-1]
VEL = findall('\d+\.\d*', NAME)[0]
string = 'CASE: '+ VEL+' SIM: '+ str(SIM_pres)+ ' PINN: '+str(PINN_pres)

print string

