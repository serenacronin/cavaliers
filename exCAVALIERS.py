#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:23:19 2023

@author: Serena A. Cronin

This script....
"""

# Import CubeFitRoutine and extra packages.
from CubeFitRoutineJanskyRun4 import CreateCube, InputParams, FitRoutine, RunFit, ModelGuesses, optical_vel_to_ang
import numpy as np
import time
from astropy.io import fits
import warnings
from tqdm import tqdm
startTime = time.time()  # TIME THE SCRIPT
warnings.filterwarnings("ignore")


# DEFINE ALL VARIABLES #
R = 3000  # MUSE resolving power
Vsys = 243.  # Koribalski+04
c = 3*10**5  # km/s
z = Vsys / c

filename = 'data/ADP.2018-11-22T21_29_46.157.fits'
wls_model = 'data/ngc253_se_halpha_vel_model_smooth_FINAL.fits'

SlabLower = 6500
SlabUpper = 6650
ContUpper1 = 6600
ContLower1 = 6545
ContUpper2 = 6750
ContLower2 = 6700
Region = False
tie_widths = True
generate_nearest_neighbor = False

# air rest wavelengths
NIIa = 6549.86 # *(1+z)
Halpha = 6564.61
NIIb = 6585.27
# SIIa = 6716.44
# SIIb = 6730.82
Voutfl = 50 # an initial guess

# amplitude guesses
amps = [100, 300, 300]  # changed to reflect the 3:1 ratio for NII

# wavelength guesses
restwls = [NIIa, Halpha, NIIb]
modelcube = fits.open(wls_model) # working with my cube now!
modelcubedat = modelcube[0].data

# vels = modelcubedat[1]
vels = modelcubedat
wls_disk = [optical_vel_to_ang(vels, restwl) for restwl in restwls]
wls = [wls_disk[0], wls_disk[1], wls_disk[2]]

# tie the center wavelengths to Halpha
tie1 = wls_disk[1] - wls_disk[0]
tie2 = wls_disk[2] - wls_disk[1]

amp_tie = 3  # the amp ratio for NII is 3:1; so NIIb = 3*NIIa

if tie_widths == True:
    ties_per_pix = [[['', 'p[4] - %f' % tie1[j,i], 'p[5]',
                      '', '', '',
                      '%f*p[0]' % amp_tie, 'p[4] + %f' % tie2[j,i], 'p[5]'] 
                     for i in range(tie1.shape[1])] 
                     for j in range(tie1.shape[0])]

elif tie_widths == False:
    ties_per_pix = [[['', 'p[4] - %f' % tie1[j,i], '',
                      '', '', '',
                      '%f*p[0]' % amp_tie, 'p[4] + %f' % tie2[j,i], ''] 
                     for i in range(tie1.shape[1])] 
                     for j in range(tie1.shape[0])]


# nested guesses and ties
n_amps = [100, 100, 300, 300, 300, 300]  # changed to reflect the 3:1 ratio for NII
n_wls = [NIIa*(Voutfl + c)/c, wls_disk[0],
          Halpha*(Voutfl + c)/c, wls_disk[1], 
          NIIb*(Voutfl + c)/c, wls_disk[2]]

if tie_widths == True:
    n_ties_per_pix = [[['', 'p[7] - %f' % tie1[j,i], 'p[8]', '', 'p[10] - %f' % tie1[j,i], 'p[11]',
                        '', '', '', '', '', '',
                        '%f*p[0]' % amp_tie, 'p[7] + %f' % tie2[j,i], 'p[8]', '%f*p[3]' % amp_tie, 'p[10] + %f' % tie2[j,i], 'p[11]'] 
                       for i in range(tie1.shape[1])] 
                       for j in range(tie1.shape[0])]

elif tie_widths == False:
    n_ties_per_pix = [[['', 'p[7] - %f' % tie1[j,i], '', '', 'p[10] - %f' % tie1[j,i], '',
                        '', '', '', '', '', '',
                        '%f*p[0]' % amp_tie, 'p[7] + %f' % tie2[j,i], '', '%f*p[3]' % amp_tie, 'p[10] + %f' % tie2[j,i], ''] 
                       for i in range(tie1.shape[1])] 
                       for j in range(tie1.shape[0])]

# starting point for the fits
point_start = (0,0)

# upper limit for the chi-square values to trigger a bad fit
free_params = [4, 10]

### --- CALL FUNCTIONS --- ####

if __name__ == '__main__':
    
    # make da cube
    cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                       ContLower2, ContUpper2, Region=Region)
        
    
    # let's run for the full cube!
    num_pix = len(cube[1,:,:][np.isfinite(cube[1,:,:])])
    # save_good_fits_num = int(round(0.01*num_pix,0))  # let's save 1% of good fits
	
    save_good_fits_num = 1
    FittingInfo = InputParams(amps, wls, R, point_start, ties=ties_per_pix,
                              nested_fit = True,
                              nested_amps = n_amps,
                              nested_wls = n_wls,
                              nested_ties = n_ties_per_pix,
                              save_failed_fits = 1,
                              savepath='fullcube-fits/JanskyRun4Test/',
                              save_good_fits = save_good_fits_num,
                              continuum_limits = [ContLower1, ContUpper2],
                              free_params = free_params,
                              random_pix_only = 300)
    
    multiprocess = 1
    RunFit(cube=cube, fitparams=FittingInfo, multiprocess=multiprocess)
    
    # TIME THE SCRIPT #
    outfile = open('fullcube-fits/JanskyRun4Test/Time_CubeFitRoutine.txt', 'w')
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime), file=outfile)
    outfile.close()
