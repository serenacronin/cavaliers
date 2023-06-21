#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
Name: run_template.py

Created on Fri Jan 13 10:23:19 2023

Author: Serena A. Cronin

This script is a template to set up and run the fitting routine.
It calls functions from routine.py and passes in initial guesses.
========================================================================================
"""

# import functions
from routine import CreateCube, InputParams, FitRoutine, optical_vel_to_ang
import time
from astropy.io import fits
import warnings
startTime = time.time()  # time the script
warnings.filterwarnings("ignore")

# ============================================================================================================ 
# VARIABLES TO CHANGE TO RUN DIFFERENT MODELS ON DIFFERENT REGIONS OF THE DATACUBE
# ============================================================================================================

fit1 = False  # set to True when you want to run the one system of lines fit
fit2 = True  # two systems of lines fit
fit3 = False  # three systems of lines
subcube = False  # do you want to work with a small region of the cube?

savepath = '../ngc253/June21/'
save_fits_num = 100

# ============================================================================================================ 
# DEFINE OTHER VARIABLES
# ============================================================================================================

R = 2989  # MUSE resolving power
Vsys = 243.  # systemic velocity of NGC 253, taken from Koribalski+04
c = 3*10**5  # speed of light in km/s
fluxnorm = 1e-20  # value that the y-axis is normalized to; taken from datacube header

# file names
filename = '../ngc253/data/ADP.2018-11-22T21_29_46.157.fits'
vel_model = '../ngc253/data/ngc253_se_halpha_vel_model_smooth_FINAL.fits'

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
Voutfl_blue = -50.0  # needed for two and three systems of lines
Voutfl_red = 50.0  # needed for three systems of lines only

# wavelength guesses
restwls = [NIIa, Halpha, NIIb, SIIa, SIIb]
modelcube = fits.open(vel_model)  # open the velocity model for the disk
modelcubedat = modelcube[0].data  # grab the data from the velocity model

if subcube == True:
    x1, x2 = 'X1 HERE', 'X2 HERE'
    y1, y2 = 'Y1 HERE', 'Y2 HERE'
    vels = modelcubedat[x1:x2, y1:y2]  # slice the velocity model to focus on a subcube/subregion
else:
	vels = modelcubedat
vels_disk = [optical_vel_to_ang(vels, Vsys, restwl) for restwl in restwls]  # convert from velocity to wavelength

# tie the center wavelengths to Halpha
tie_niia = Halpha - NIIa
tie_niib = NIIb - Halpha
tie_siia = SIIa - Halpha
tie_siib = SIIb - Halpha

# the amp ratio for NII is 3:1; so NIIb = 3*NIIa
amp_tie = 3

# parameters we are allowing to float (i.e., not tied)
if fit1 == True:
     free_params = 6
elif fit2 == True:
     free_params = 12
elif fit3 == True:
     free_params = 18


# ============================================================================================================ 
# ONE SYSTEM OF LINES
# ============================================================================================================

if fit1 == True:

    # amplitude + wavelength guesses
    amps1 = [100, 300, 300, 100, 120]  # 3:1 ratio for NII
    wls1 = [vels_disk[0], vels_disk[1], vels_disk[2], vels_disk[3], vels_disk[4]]

    # tying amps, wavelengths, widths
    ties1_per_pix = ['', '%f * p[4] / %f' % (NIIa, Halpha), 'p[5]',
                    '', '', '',
                    '%f*p[0]' % amp_tie, '%f * p[4] / %f' % (NIIb, Halpha), 'p[5]',
                    '', '%f * p[4] / %f' % (SIIa, Halpha), 'p[5]',
                    '', '%f * p[4] / %f' % (SIIb, Halpha), 'p[5]']
else:
    amps1 = False
    wls1 = False
    ties1_per_pix = False

# ============================================================================================================ 
# TWO SYSTEMS OF LINES
# ============================================================================================================

if fit2 == True:

    # amplitude + wavelength guesses
	# first wavelength guess is blueshifted with respect to the disk guess
    amps2 = [100, 100, 300, 300, 300, 300, 100, 100, 150, 150]
    wls2 = [vels_disk[0]*(Voutfl_blue + c)/c, vels_disk[0],
            vels_disk[1]*(Voutfl_blue + c)/c, vels_disk[1],
            vels_disk[2]*(Voutfl_blue + c)/c, vels_disk[2],
            vels_disk[3]*(Voutfl_blue + c)/c, vels_disk[3],
            vels_disk[4]*(Voutfl_blue + c)/c, vels_disk[4]]

    # tying amps, wavelengths, widths
    ties2_per_pix = ['', '%f * p[7] / %f' % (NIIa, Halpha), 'p[8]', '', '%f * p[10] / %f' % (NIIa, Halpha), 'p[11]',
                        '', '', '', '', '', '',
                        '%f*p[0]' % amp_tie, '%f * p[7] / %f' % (NIIb, Halpha), 'p[8]', '%f*p[3]' % amp_tie, '%f * p[10] / %f' % (NIIb, Halpha), 'p[11]',
                        '', '%f * p[7] / %f' % (SIIa, Halpha), 'p[8]', '', '%f * p[10] / %f' % (SIIa, Halpha), 'p[11]',
                        '', '%f * p[7] / %f' % (SIIb, Halpha), 'p[8]', '', '%f * p[10] / %f' % (SIIb, Halpha), 'p[11]']

else:
    amps2 = False
    wls2 = False
    ties2_per_pix = False

# ============================================================================================================ 
# THREE SYSTEMS OF LINES
# ============================================================================================================

if fit3 == True:

    # amplitude + wavelength guesses
	# first wavelength guess is blueshifted with respect to the disk guess
	# third wavelength guess is redshifted with respect to the disk guess
    amps3 = [100, 100, 100, 300, 300, 300, 300, 300, 300, 100, 100, 100, 150, 150, 150]
    wls3 = [vels_disk[0]*(Voutfl_blue + c)/c, vels_disk[0], vels_disk[0]*(Voutfl_red + c)/c,
            vels_disk[1]*(Voutfl_blue + c)/c, vels_disk[1], vels_disk[1]*(Voutfl_red + c)/c,
            vels_disk[2]*(Voutfl_blue + c)/c, vels_disk[2], vels_disk[2]*(Voutfl_red + c)/c,
            vels_disk[3]*(Voutfl_blue + c)/c, vels_disk[3], vels_disk[3]*(Voutfl_red + c)/c,
            vels_disk[4]*(Voutfl_blue + c)/c, vels_disk[4], vels_disk[4]*(Voutfl_red + c)/c]

    # tying amps, wavelengths, widths
    ties3_per_pix = ['', '%f * p[10] / %f' % (NIIa, Halpha), 'p[11]', '', '%f * p[13] / %f' % (NIIa, Halpha), 'p[14]', '', '%f * p[16] / %f' % (NIIa, Halpha), 'p[17]', 
                    '', '', '', '', '', '', '', '', '',
                    '%f*p[0]' % amp_tie, '%f * p[10] / %f' % (NIIb, Halpha), 'p[11]', '%f*p[6]' % amp_tie, '%f * p[13] / %f' % (NIIb, Halpha), 'p[14]', '%f*p[6]' % amp_tie, '%f * p[16] / %f' % (NIIb, Halpha), 'p[17]',
                    '', '%f * p[10] / %f' % (SIIa, Halpha), 'p[11]', '', '%f * p[13] / %f' % (SIIa, Halpha), 'p[14]', '', '%f * p[16] / %f' % (SIIa, Halpha), 'p[17]',
                    '', '%f * p[10] / %f' % (SIIb, Halpha), 'p[11]', '', '%f * p[13] / %f' % (SIIb, Halpha), 'p[14]', '', '%f * p[16] / %f' % (SIIb, Halpha), 'p[17]']
else:
    amps3 = False
    wls3 = False
    ties3_per_pix = False
    

# ============================================================================================================ 
# MAIN FUNCTION
# ============================================================================================================

if __name__ == '__main__':
    
    # make the cube
    cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1, ContLower2, ContUpper2)
    if subcube == True:
         cube = cube[:,y1:y2, x1:x2]  # slicing is in the format of z, y, x
	
    FittingInfo = InputParams(fit1, fit2, fit3, R, free_params, 
                            continuum_limits=[ContLower1, ContUpper2], fluxnorm=fluxnorm,
                            amps1=amps1, centers1=wls1, ties1=ties1_per_pix,
                            amps2=amps2, centers2=wls2, ties2=ties2_per_pix,
                            amps3=amps3, centers3=wls3, ties3=ties3_per_pix,
                            save_fits=save_fits_num, savepath=savepath)
    
    # time the fitting routine so we know how long it takes to run
    if fit1 == True:
          fitnum = '1'
    elif fit2 == True:
          fitnum = '2'
    elif fit3 == True:
         fitnum = '3'

    FitRoutine(FittingInfo=FittingInfo, cube=cube)

    outfile = open('%sTime_CubeFitRoutine_Fit%s.txt'% (savepath, fitnum), 'w')
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))  # prints to the terminal
    print('Execution time in seconds: ' + str(executionTime), file=outfile)  # prints to a file
    outfile.close()
