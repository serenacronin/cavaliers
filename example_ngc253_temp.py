#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:38:09 2022

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
from CubeFitRoutineV6 import CreateCube, InputParamsCenterModel, FitRoutine, RunFit, ModelGuesses, optical_vel_to_ang
import time
from astropy.io import fits
import warnings
from reproject import reproject_interp
startTime = time.time()  # TIME THE SCRIPT
warnings.filterwarnings("ignore")

# DEFINE ALL VARIABLES #
R = 3000  # MUSE resolving power
Vsys = 243.  # Koribalski+04
c = 3*10**5  # km/s
z = Vsys / c
multiprocess = 1

filename = 'data/ADP.2018-11-22T21_29_46.157.fits'
wls_model = 'testing_reproject.fits'
# wls_model = 'data/ngc253_se_halpha_vel_model.fits'
# wls_model = 'data/ngc253_se_halpha_vel_model_region1.fits'
SlabLower = 6500
SlabUpper = 6650
ContUpper1 = 6600
ContLower1 = 6545
ContUpper2 = 6750
ContLower2 = 6700
# Region = 'data/ngc253_se_subcube.reg'
Region = False

# air rest wavelengths
NIIa = 6549.86 # *(1+z)
Halpha = 6564.61
NIIb = 6585.27
# SIIa = 6716.44
# SIIb = 6730.82
Voutfl = 200 # an initial guess

# amplitude guesses
amps = [100, 300, 450]

# wavelength guesses
restwls = [NIIa, Halpha, NIIb]
modelcube = fits.open(wls_model) # working with my cube now!
modelcubedat = modelcube[0].data
vels = modelcubedat
wls_disk = [optical_vel_to_ang(vels, restwl) for restwl in restwls]
wls = [wls_disk[0], wls_disk[1], wls_disk[2]]

# tie the center wavelengths to Halpha
# tie1 = Halpha - NIIa
# tie2 = NIIb - Halpha

tie1 = wls_disk[1] - wls_disk[0]
tie2 = wls_disk[2] - wls_disk[1]


ties_per_pix = [[['', 'p[4] - %f' % tie1[j,i], '',
                 '', '', '',
                 '', 'p[4] + %f' % tie2[j,i], ''] for j in range(tie1.shape[0])] for i in range(tie1.shape[1])]

# ties = ['', 'p[4] - %f' % tie1, '',
#         '', '', '',
#         '', 'p[4] + %f' % tie2, '']

# nested guesses and ties
n_amps = [100, 100, 300, 300, 450, 450]
n_wls = [NIIa*(Voutfl + c)/c, wls_disk[0],
          Halpha*(Voutfl + c)/c, wls_disk[1], 
          NIIb*(Voutfl + c)/c, wls_disk[2]]
# n_ties = ['', 'p[7] - %f' % tie1, '', '', 'p[10] - %f' % tie1, '',
#           '', '', '', '', '', '',
#           '', 'p[7] + %f' % tie2, '', '', 'p[10] + %f' % tie2, '']


n_ties_per_pix = [[['', 'p[7] - %f' % tie1[j,i], '', '', 'p[10] - %f' % tie1[j,i], '',
          '', '', '', '', '', '',
          '', 'p[7] + %f' % tie2[j,i], '', '', 'p[10] + %f' % tie2[j,i], ''] for j in range(tie1.shape[0])] for i in range(tie1.shape[1])]


# starting point for the fits
point_start = (0,0)

# upper limit for the chi-square values to trigger a bad fit
# doing a chi-square probability of 0.999999
free_params = [7, 16]

# CALL FUNCTIONS #

# make da cube
cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                   ContLower2, ContUpper2, Region=Region)

# generate the input params
FittingInfo = InputParamsCenterModel(amps, wls, R, point_start, ties=ties_per_pix,
                                     nested_fit=True,
                                     nested_amps=n_amps,
                                     nested_wls=n_wls,
                                     nested_ties=n_ties_per_pix,
                                     failed_fit=True,
                                     # savepath='subcube-fits/subcube2.0/region1_HaModel/',
                                     savepath='fullcube-fits/nosii-CFRv6/',
                                     save_good_fits=True,
                                     continuum_limits=[ContLower1, ContUpper2],
                                     free_params = free_params,
                                     nearest_neighbor = True)

# (256, 8)

RunFit(cube=cube, fitparams=FittingInfo, multiprocess=multiprocess)

# TIME THE SCRIPT #
outfile = open('fullcube-fits/nosii-CFRv6/Time_CubeFitRoutine.txt', 'w')
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime), file=outfile)
outfile.close()
