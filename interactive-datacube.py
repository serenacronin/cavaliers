#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:18:50 2022

@author: Serena A. Cronin

This script allows me to interact with my ratio plots!
I can now pull out a spectrum in each pixel with the click of a button!

"""

### ---- MY FUNCTIONS ---- ###
import sys
sys.path.append('../astro_tools')

from gauss_tools import one_gaussian, two_gaussian
from CubeFitRoutineJanskyRun2 import CreateCube

### ---- OTHER FUNCTIONS & PACKAGES ---- ###
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.lines import Line2D
from astropy.wcs import wcs
import time
startTime = time.time()  # TIME THE SCRIPT

##### --- DEFINE FUNCTIONS --- #####

def onpick(event):
    
    """
    
    Function that determines what to do when the user clicks
    the plot!
    
    event : Class
        Stores the event information (e.g., a mouse click)
    ParCubeFile : str
        File string of the data cube that contains the fit parameter info
        needed for the pop-up plots.
    
    """
    
    ParCubeFile = 'JanskyRun3/parcube_final.fits'
    
    # spectral cube for extracting the spectrum per pixel
    # variables for my CreateCube function
    SlabLower = 6500
    SlabUpper = 6650
    ContUpper1 = 6600
    ContLower1 = 6545
    ContUpper2 = 6750
    ContLower2 = 6700
    Region = False
    SpecCube = CreateCube(OrigCubeFile, SlabLower, SlabUpper, ContLower1, ContUpper1,
                        ContLower2, ContUpper2, Region=Region)
    
    # open the parameter cube fits file
    cube = fits.open(ParCubeFile)[0]
    data = cube.data
    npars, y, x = cube.shape
    
    
    # THE ORDER OF IX, IY IS CORRECT FOR THE NEXT SEVERAL LINES
    # IDK WHY IT IS DOING THIS I GUESS EVENT SWAPPED THE AXES
    # IDK
    # JUST DON'T CHANGE THEM A;LSDKFKL;JASDL;KFJKSLA
    # get the coordinates of the event
    ix, iy = event.xdata, event.ydata

    # get the parameter info at those coordinates  
    params = cube.data[:,int(round(ix,0)),int(round(iy,0))]
    
    # plot the actual spectrum from the spectral cube
    # get the x and y-axis data
    spectrum = np.array(SpecCube[:,int(round(ix,0)),int(round(iy,0))], dtype='float64')  # y-axis data
    minval = min(np.array(SpecCube.spectral_axis))
    maxval = max(np.array(SpecCube.spectral_axis))
    x_data = np.linspace(minval, maxval, len(spectrum)) # x-axis data
    
    # re-create the composite NIIa gaussian using the parameter info
    NIIa_tot = two_gaussian(x_data, 
                  params[0], params[1], 
                  params[2],params[3], 
                  params[4], params[5])
    
    # re-create the individual component NIIa gaussians using the par info
    NIIa_comp1 = one_gaussian(x_data, 
                  params[0], params[1], 
                  params[2])
    
    NIIa_comp2 = one_gaussian(x_data, 
                 params[3], params[4], 
                 params[5])
    
    # re-create the composite H-alpha gaussian using the parameter info
    Halpha_tot = two_gaussian(x_data, 
                  params[6], params[7], 
                  params[8],params[9], 
                  params[10], params[11])
    
    # re-create the individual component H-alpha gaussians using the par info
    Halpha_comp1 = one_gaussian(x_data, 
                  params[6], params[7], 
                  params[8])
    
    Halpha_comp2 = one_gaussian(x_data, 
                  params[9], params[10], 
                  params[11])
    
    
    # re-create the composite NIIb gaussian using the parameter info
    NIIb_tot = two_gaussian(x_data, 
                  params[12], params[13], 
                  params[14], params[15], 
                  params[16], params[17])
    
    # re-create the individual component NIIb gaussians using the par info
    NIIb_comp1 = one_gaussian(x_data, 
                  params[12], params[13], 
                  params[14])
    
    NIIb_comp2 = one_gaussian(x_data, 
                  params[15], params[16], 
                  params[17])
    
    # re-create the composite NIIb gaussian using the parameter info
    SIIa_tot = two_gaussian(x_data, 
                  params[12], params[13], 
                  params[14], params[15], 
                  params[16], params[17])
    
    # re-create the individual component H-alpha gaussians using the par info
    NIIb_comp1 = one_gaussian(x_data, 
                  params[12], params[13], 
                  params[14])
    
    NIIb_comp2 = one_gaussian(x_data, 
                  params[15], params[16], 
                  params[17])

    # make the plot!
    fig, ax = plt.subplots()
    
    # plot the spectrum
    ax.plot(x_data, spectrum, color='gray', alpha=0.5, lw=2)
    
    # plot the fits
    ax.plot(x_data, NIIa_tot, color = 'tab:pink', lw=2)    
    ax.plot(x_data, NIIa_comp1, color = 'tab:cyan', lw=1) 
    ax.plot(x_data, NIIa_comp2, color = 'tab:cyan', lw=1) 
    
    ax.plot(x_data, Halpha_tot, color='tab:pink', lw=2)
    ax.plot(x_data, Halpha_comp1, color = 'tab:cyan', lw=1) 
    ax.plot(x_data, Halpha_comp2, color = 'tab:cyan', lw=1) 
    
    ax.plot(x_data, NIIb_tot, color='tab:pink', lw=2)
    ax.plot(x_data, NIIb_comp1, color = 'tab:cyan', lw=1) 
    ax.plot(x_data, NIIb_comp2, color = 'tab:cyan', lw=1) 
    
    # set axes labels
    ax.tick_params(axis='both', which='both',direction='in',
                   width=2.5, labelsize=16, length=7)
    ax.set_xlabel('Wavelength ($\AA$)', fontsize=20)
    ax.set_ylabel(r'Flux $(10^{-20} \mathrm{erg/s/cm^2/\AA})$', fontsize=20)
    
    # set a title
    ax.set_title('Pixel: (%s, %s)' % (int(round(ix,0)),int(round(iy,0))), fontsize=20)
    
    # plot a legend
    custom_lines = [Line2D([0], [0], color='tab:pink', lw=2),
                    Line2D([0], [0], color='tab:cyan', lw=2)]
    plt.legend(custom_lines,['Composite', 'Components'], fontsize=16, 
              loc='upper left')
    
    fig.show()

##### --- SET UP PLOT --- #####
fig = plt.figure(figsize=(20,20))
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.variant"] = "small-caps"

##### --- DEFINE VARIABLES --- #####

# par = 3
par = 4
CubeFile = 'JanskyRun3/ratio_cube.fits'
ParCubeFile = 'JanskyRun3/parcube_final.fits'
OrigCubeFile = 'data/ADP.2018-11-22T21_29_46.157.fits'
vmin = 0
vmax = 6
cmap = 'jet'
label = 'peak flux'
# title = 'NIIb / H-alpha (Wind)'
title = 'NIIb / NIIa (Disk)'

# variables for my CreateCube function
SlabLower = 6500
SlabUpper = 6650
ContUpper1 = 6600
ContLower1 = 6545
ContUpper2 = 6750
ContLower2 = 6700
Region = False

##### --- MAIN SCRIPT --- #####

# spectral cube for extracting the spectrum per pixel
SpecCube = CreateCube(OrigCubeFile, SlabLower, SlabUpper, ContLower1, ContUpper1,
                    ContLower2, ContUpper2, Region=Region)

# fits data cube file
cube = fits.open(CubeFile)[0]
hdr = cube.header
data = cube.data
npars, y, x = cube.shape
w = wcs.WCS(hdr)

# plot the map
# fig, ax = plt.subplots(1,1, subplot_kw={'projection': w[0]})
fig, ax = plt.subplots(1,1)
im = ax.imshow(data[par], vmin=vmin, vmax=vmax, 
                origin='lower', cmap=cmap, picker=True)

ax.tick_params(axis='both', which='both',direction='in',
               width=2.5, labelsize=16, length=7)
ax.set_xlabel('x', fontsize=20)
ax.set_ylabel('y', fontsize=20)

bar = plt.colorbar(im, fraction=0.046)
bar.set_label(label, fontsize=18)
bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
ax.set_title(title, fontsize=20)

fig.canvas.mpl_connect('button_press_event', onpick)
plt.show()

##### --- TIME THE SCRIPT --- #####
outfile = open('Time_.txt', 'w')
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
outfile.close()