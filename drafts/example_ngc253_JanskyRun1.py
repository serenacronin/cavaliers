##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:54:28 2022

@author: Serena A. Cronin

Test run of the full cube on jansky using the Halpha map as an initial guess
and tying the NII amps and widths together! This is the first run of 
parallelization using the Halpha map.

"""

# Import CubeFitRoutine and extra packages.
from CubeFitRoutineJanskyRun1 import CreateCube, InputParams, FitRoutine, RunFit, ModelGuesses, optical_vel_to_ang
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
    
    
    # option to generate nearest neighbor guesses
    # this will choose a random subset of pixels, fit them, and then
    # create a map of the outflow answers to then use on the whole cube
    if generate_nearest_neighbor == True:
        
        nearest_neighbor = False  # false because we're currently generating 
                                  # the nearest neighbor guesses!
        
        # how many pixels are we dealing with? (not nan)
        num_pix = len(cube[1,:,:][np.isfinite(cube[1,:,:])])
        random_pix_num = int(round(0.01*num_pix,0))  # let's use 1% of pixels
        
        # probably best to save all the fits!
        save_good_fits = 1
        save_failed_fits = 1
        
        # generate the input params
        FittingInfo = InputParams(amps, wls, R, point_start, ties=ties_per_pix,
                                  nested_fit = True,
                                  nested_amps = n_amps,
                                  nested_wls = n_wls,
                                  nested_ties = n_ties_per_pix,
                                  save_failed_fits = save_failed_fits,
                                  savepath='trial-run-no-multiprocess/',
                                  save_good_fits = save_good_fits,
                                  continuum_limits = [ContLower1, ContUpper2],
                                  free_params = free_params,
                                  nearest_neighbor = nearest_neighbor,
                                  random_pix_only = random_pix_num)
        
        multiprocess = 1  # no need to parallelize with so few pixels
        RunFit(cube=cube, fitparams=FittingInfo, multiprocess=multiprocess)
        
        # TIME THE SCRIPT #
        outfile = open('trial-run-no-multiprocess/Time_CubeFitRoutine.txt', 'w')
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime), file=outfile)
        outfile.close()
        
    
    # now let's run for the full cube!
    num_pix = len(cube[1,:,:][np.isfinite(cube[1,:,:])])
    save_good_fits_num = int(round(0.01*num_pix,0))  # let's save 1% of good fits
    FittingInfo = InputParams(amps, wls, R, point_start, ties=ties_per_pix,
                              nested_fit = True,
                              nested_amps = n_amps,
                              nested_wls = n_wls,
                              nested_ties = n_ties_per_pix,
                              save_failed_fits = 1,
                              savepath='trial-run-no-multiprocess/',
                              save_good_fits = save_good_fits_num,
                              continuum_limits = [ContLower1, ContUpper2],
                              free_params = free_params,
                              nearest_neighbor = True,
                              random_pix_only = False)
    
    multiprocess = 1
    RunFit(cube=cube, fitparams=FittingInfo, multiprocess=multiprocess)
    
    # TIME THE SCRIPT #
    outfile = open('trial-run-no-multiprocess/Time_CubeFitRoutine.txt', 'w')
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime), file=outfile)
    outfile.close()
