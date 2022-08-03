#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:40:25 2022

@author: serenac
"""

"""

This is an example of fitting IFS data. Here we have MUSE data cubes
of the central region of NGC 253, a nearby starburst galaxy. We want to fit 
Gaussian profiles to [NII], HAlpha, and [SII] emission lines. These lines are
either single or doubly-peaked, the latter of which occurs when we have 
galactic winds on top of galactic rotation. The functions in CubeFitRoutine.py
were developed to solve this fitting puzzle and then generalized to be used
in many other cases of fitting Gaussians to entire data cubes.

CubeFitRoutine.py already imports numpy, pyspeckit, spectral-cube, astropy,
regions, Pool, and partial. All of these packages are required. You will also 
need my gauss_tools.py package, or just create a single gaussian function 
called one_gaussian and a reduced chi square function called red_chisq.

"""

# Import CubeFitRoutine and extra packages.
from CubeFitRoutineV4 import CreateCube, InputParams, FitRoutine, RunFit
import time
import warnings
startTime = time.time()  # TIME THE SCRIPT
warnings.filterwarnings("ignore")

# DEFINE ALL VARIABLES #
R = 3000  # MUSE resolving power
Vsys = 243.  # Koribalski+04
c = 3*10**5  # km/s
z = Vsys / c
multiprocess = 1

filename = 'data/ADP.2018-11-22T21_29_46.157.fits'
SlabLower = 6500
SlabUpper = 6650
ContUpper1 = 6600
ContLower1 = 6545
ContUpper2 = 6750
ContLower2 = 6700
# Region = 'data/ngc253_subcube2.reg'
Region = False

# air rest wavelengths
NIIa = 6549.86 # *(1+z)
Halpha = 6564.61
NIIb = 6585.27
# SIIa = 6716.44
# SIIb = 6730.82
Voutfl = 50 # an initial guess

# wavelength, amplitude guesses
wls = [NIIa*(Vsys + c)/c, Halpha*(Vsys + c)/c, NIIb*(Vsys + c)/c]
amps = [100, 300, 450]

# tie the center wavelengths to Halpha
tie1 = Halpha - NIIa
tie2 = NIIb - Halpha

ties = ['', 'p[4] - %f' % tie1, '',
        '', '', '',
        '', 'p[4] + %f' % tie2, '']

# nested guesses and ties
n_wls = [NIIa*(Voutfl + c)/c, NIIa*(Vsys + c)/c, Halpha*(Voutfl + c)/c, 
         Halpha*(Vsys + c)/c, NIIb*(Voutfl + c)/c, NIIb*(Vsys + c)/c]
n_amps = [100, 100, 300, 300, 450, 450]
n_ties = ['', 'p[7] - %f' % tie1, '', '', 'p[10] - %f' % tie1, '',
          '', '', '', '', '', '',
          '', 'p[7] + %f' % tie2, '', '', 'p[10] + %f' % tie2, '']

# starting point for the fits
point_start = (0,0)
# (256, 8)

# upper limit for the chi-square values to trigger a bad fit
# doing a chi-square probability of 0.999999
free_params = [7, 16]

# CALL FUNCTIONS #

# make da cube
cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                   ContLower2, ContUpper2, Region=Region)

# generate the input params
FittingInfo = InputParams(amps, wls, R, point_start, ties,
                          nested_fit=True,
                          nested_amps=n_amps,
                          nested_wls=n_wls,
                          nested_ties=n_ties,
                          failed_fit=True,
                          savepath='fullcube-fits/nosii-CFRv6/',
                          save_good_fits=True,
                          continuum_limits=[ContLower1, ContUpper2],
                          free_params = free_params)

# run the fits!
RunFit(cube=cube, fitparams=FittingInfo, multiprocess=multiprocess)

# TIME THE SCRIPT #
outfile = open('fullcube-fits/nosii-CFRv6/Time_CubeFitRoutine.txt', 'w')
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime), file=outfile)
outfile.close()
