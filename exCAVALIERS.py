#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:23:19 2023

@author: Serena A. Cronin

This script....
"""

# Import CubeFitRoutine and extra packages.
from CAVALIERS import CreateCube, InputParams, RunFit, optical_vel_to_ang
import numpy as np
import time
from astropy.io import fits
import warnings
startTime = time.time()  # TIME THE SCRIPT
warnings.filterwarnings("ignore")


# DEFINE ALL VARIABLES #
R = 3000  # MUSE resolving power
Vsys = 243.  # Koribalski+04
c = 3*10**5  # km/s
z = Vsys / c

# file names
filename = '../ngc253/data/ADP.2018-11-22T21_29_46.157.fits'
wls_model = '../ngc253/data/ngc253_se_halpha_vel_model_smooth_FINAL.fits'
# filename = '../ADP.2018-11-22T21_29_46.157.fits'
# wls_model = '../ngc253_se_halpha_vel_model_smooth_FINAL.fits'

# info for continuum
SlabLower = 6500
SlabUpper = 6800
ContUpper1 = 6620
ContLower1 = 6525
ContUpper2 = 6750
ContLower2 = 6700

# air rest wavelengths
NIIa = 6548.05
Halpha = 6562.801
NIIb = 6583.45
SIIa = 6716.44
SIIb = 6730.82
Voutfl_blue = -50.0 # an initial guess
Voutfl_red = 50.0 # an initial guess

# wavelength guesses
restwls = [NIIa, Halpha, NIIb, SIIa, SIIb]
modelcube = fits.open(wls_model)
modelcubedat = modelcube[0].data
vels = modelcubedat
wls_disk = [optical_vel_to_ang(vels, restwl) for restwl in restwls]

# tie the center wavelengths to Halpha
tie_niia = Halpha - NIIa
tie_niib = NIIb - Halpha
tie_siia = SIIa - Halpha
tie_siib = SIIb - Halpha

# the amp ratio for NII is 3:1; so NIIb = 3*NIIa
amp_tie = 3

# parameters we are allowing to float (i.e., not tied)
free_params = [6, 12, 18]

## buttons to toggle ##
Region = False
fit1 = True
fit2 = True
fit3 = True
rand_pix_num = 100
# redchisq_range = '6525:6620, 6700:6750'
redchisq_range = np.array([np.arange(6525,6620), np.arange(6700,6750)])
# rand_pix_num = False
savepath = '../ngc253/testFeb9/'
# savepath = 'testFeb6/'
multiprocess = 1
save_fits_num = 1

# num_pix = len(cube[1,:,:][np.isfinite(cube[1,:,:])])
# save_good_fits_num = int(round(0.01*num_pix,0))  # let's save 1% of good fits

############################# 
##### ONE COMPONENT FIT #####
#############################

if fit1 == True:

    # amplitude + wavelength guesses
    amps1 = [100, 300, 300, 100, 120]  # 3:1 ratio for NII
    wls1 = [wls_disk[0], wls_disk[1], wls_disk[2], wls_disk[3], wls_disk[4]]

    # tying amps, wavelengths, widths
    ties1_per_pix = ['', 'p[4] - %f' % tie_niia, 'p[5]',
                    '', '', '',
                    '%f*p[0]' % amp_tie, 'p[4] + %f' % tie_niib, 'p[5]',
                    '', 'p[4] + %f' % tie_siia, 'p[5]',
                    '', 'p[4] + %f' % tie_siib, 'p[5]']

############################# 
##### TWO COMPONENT FIT #####
#############################

if fit2 == True:

    # amplitude + wavelength guesses
    amps2 = [100, 100, 300, 300, 300, 300, 100, 100, 150, 150]
    wls2 = [NIIa*(Voutfl_blue + c)/c, wls_disk[0],
            Halpha*(Voutfl_blue + c)/c, wls_disk[1],
            NIIb*(Voutfl_blue + c)/c, wls_disk[2],
            SIIa*(Voutfl_blue + c)/c, wls_disk[3],
            SIIb*(Voutfl_blue + c)/c, wls_disk[4]]

    # tying amps, wavelengths, widths
    ties2_per_pix = ['', 'p[7] - %f' % tie_niia, 'p[8]', '', 'p[10] - %f' % tie_niia, 'p[11]',
                        '', '', '', '', '', '',
                        '%f*p[0]' % amp_tie, 'p[7] + %f' % tie_niib, 'p[8]', '%f*p[3]' % amp_tie, 'p[10] + %f' % tie_niib, 'p[11]',
                        '', 'p[7] + %f' % tie_siia, 'p[8]', '', 'p[10] + %f' % tie_siia, 'p[11]',
                        '', 'p[7] + %f' % tie_siib, 'p[8]', '', 'p[10] + %f' % tie_siib, 'p[11]']

############################# 
#### THREE COMPONENT FIT ####
#############################

if fit3 == True:

    # amplitude + wavelength guesses
    amps3 = [100, 100, 100, 300, 300, 300, 300, 300, 300, 100, 100, 100, 150, 150, 150]
    wls3 = [NIIa*(Voutfl_blue + c)/c, wls_disk[0], NIIa*(Voutfl_red + c)/c,
            Halpha*(Voutfl_blue + c)/c, wls_disk[1], Halpha*(Voutfl_red + c)/c,
            NIIb*(Voutfl_blue + c)/c, wls_disk[2], NIIb*(Voutfl_red + c)/c,
            SIIa*(Voutfl_blue + c)/c, wls_disk[3], SIIa*(Voutfl_red + c)/c,
            SIIb*(Voutfl_blue + c)/c, wls_disk[4], SIIb*(Voutfl_red + c)/c]

    # tying amps, wavelengths, widths
    ties3_per_pix = ['', 'p[10] - %f' % tie_niia, 'p[11]', '', 'p[13] - %f' % tie_niia, 'p[14]', '', 'p[16] - %f' % tie_niia, 'p[17]', 
                    '', '', '', '', '', '', '', '', '',
                    '%f*p[0]' % amp_tie, 'p[10] + %f' % tie_niib, 'p[11]', '%f*p[6]' % amp_tie, 'p[13] + %f' % tie_niib, 'p[14]', '%f*p[6]' % amp_tie, 'p[16] + %f' % tie_niib, 'p[17]',
                    '', 'p[10] + %f' % tie_siia, 'p[11]', '', 'p[13] + %f' % tie_siia, 'p[14]', '', 'p[16] + %f' % tie_siia, 'p[17]',
                    '', 'p[10] + %f' % tie_siib, 'p[11]', '', 'p[13] + %f' % tie_siib, 'p[14]', '', 'p[16] + %f' % tie_siib, 'p[17]']


### --- CALL FUNCTIONS --- ####

if __name__ == '__main__':
    
    # make da cube
    cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                       ContLower2, ContUpper2, Region=Region)
        
    
    # let's run for the full cube!
    # num_pix = len(cube[1,:,:][np.isfinite(cube[1,:,:])])
    # save_good_fits_num = int(round(0.01*num_pix,0))  # let's save 1% of good fits
	
    FittingInfo = InputParams(fit1, fit2, fit3, R, free_params, 
                            continuum_limits=[ContLower1, ContUpper2],
                            amps1=amps1, centers1=wls1, ties1=ties1_per_pix,
                            amps2=amps2, centers2=wls2, ties2=ties2_per_pix,
                            amps3=amps3, centers3=wls3, ties3=ties3_per_pix,
                            redchisq_range=redchisq_range, random_pix_only=rand_pix_num, 
                            save_fits=save_fits_num, savepath=savepath)
    

    RunFit(cube=cube, fitparams=FittingInfo, multiprocess=multiprocess)
    
    # TIME THE SCRIPT #
    outfile = open('%sTime_CubeFitRoutine.txt'% savepath, 'w')
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    print('Execution time in seconds: ' + str(executionTime), file=outfile)
    outfile.close()
