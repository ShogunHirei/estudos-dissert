"""
File: outputPressRPD.py
Author: ShogunHirei
Description: Paraview script to output the mean of the Relative Percentual 
             Difference (RPD). 

Notes: The `PVFoamReader` is MANDATORY! Remember to use the LD_PRELOAD
        such as described in: https://discourse.paraview.org/t/not-able-to-load-openfoam-paraview-reader-pvfoamreader-when-using-pvbatch-or-pvpython/248/12
"""

import re
import argparse as ap

# Inputs definition

parser = ap.ArgumentParser()
# Case to read
parser.add_argument("case", help='Case that will have the mean calculated')

# FoamFile to read
parser.add_argument("foamfile", help='The path to the .OpenFOAM file to read')

# File to output response (.csv)
parser.add_argument("outfile", help='Output file that will be group the data')

args = parser.parse_args()


CASE = args.case
FOAMFILE = args.foamfile
OUTFILE = args.outfile

from paraview.simple import *

# Reading the Foam file
BODY = PVFoamReader(FileName=FOAMFILE)

# Adjusting Volume Fields
BODY.VolumeFields = ['p', 'ANN_p']

# Output of script
output = {}

# Values: Inlet, Mean(RPD_all), Mean(RPD_Outliers_ClippedOut)
output['Inlet'] = float(re.findall('\d*\.\d*$', CASE)[0])

print(CASE, FOAMFILE, OUTFILE)

# # get last available time
# Show()
# CAM = GetActiveView()
# CAM.ViewTime = BODY.TimestepValues[-1]


# Applying Calculator Filter
RPD = Calculator(Input=BODY)
RPD.ResultArrayName = 'RPD'
RPD.AttributeType = 'Cell Data'
RPD.Function = 'abs((ANN_p - p)/p)*100'
RPD.UpdatePipeline()

# Second value: mean value with outliers
pyCalc = PythonCalculator(Input=RPD)
pyCalc.ArrayAssociation = 'Cell Data'
pyCalc.ArrayName = 'mean'
pyCalc.Expression = 'mean(inputs[0].CellData["RPD"])'
pyCalc.UpdatePipeline()

# Output value
output['RPD_All'] = pyCalc.CellData.GetArray('mean').GetRange()[0]

# Removing outliers with Clip filter
CP = Clip(Input=RPD)
CP.ClipType = 'Scalar'
CP.Scalars = 'RPD'
# Values higher than 100% of deviation are considered outliers
CP.Value = 75
CP.UpdatePipeline()

# Calculating the mean with the PythonCalculator filter
# Changing Input to reuse the filter
pyCalc.Input = CP
pyCalc.UpdatePipeline()

# Output value
output['RPD_Less'] = pyCalc.CellData.GetArray('mean').GetRange()[0]

print('Values achieved!')
print(output)

# Writing to output file (considering append to end of file)
with open(OUTFILE, 'a') as fn:
    # Writing output
    fn.write(str(round(output['Inlet'], 4)) + ',')
    fn.write(str(round(output['RPD_All'], 4)) + ',')
    fn.write(str(round(output['RPD_Less'], 4)) + '\n')

print('Finished!')





