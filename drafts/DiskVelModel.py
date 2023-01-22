#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:03:27 2022

@author: serenac
"""

"""

The brightest points of Halpha should come from the disk of the galaxy.
Therefore, by fitting a parabola to the brightest Halpha peak in each
pixel and extracting the velocity (i.e., the vertex), 
we can theoretically construct a map of the disk.

"""

from CubeFitRoutine import CreateCube
import numpy as np
import pyspeckit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy import units as u
from astropy.io import fits
from astropy.wcs import wcs
import time
startTime = time.time()  # TIME THE SCRIPT

### ---- DEFINE FUNCTIONS AND VARIABLES ---- ###

def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    
		'''
		http://chris35wills.github.io/parabola_python/
        
		'''

		denom = (x1-x2) * (x1-x3) * (x2-x3)
		A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
		B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
		C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

		return A,B,C
    
    
def optical_vel_to_ang(vels, restwl):
    
    c = 3.0*10**5
    wls = restwl*(vels + c)/c
    
    return wls


def ang_to_optical_vel(wls, restwl):
    
    c = 3.0*10**5
    vels = c*((wls/restwl)-1)
    
    return vels

def secax_fxn(xvals):
    c = 3.0*10**5
    Vsys = 243.
    Halpha = 6564.61*(Vsys + c)/c
    return Halpha*(xvals + c)/c

def secax_fxn_inverse(xvals):
    Vsys = 243.
    c = 3.0*10**5
    Halpha = 6564.61*(Vsys + c)/c
    return c*((xvals/Halpha)-1)

filename = 'data/ADP.2018-11-22T21_29_46.157.fits'
SlabLower = 6500
SlabUpper = 6650
ContUpper1 = 6600
ContLower1 = 6545
ContUpper2 = 6750
ContLower2 = 6700
# Region = 'data/ngc253_se_subcube.reg'
# Region = 'data/ha_map_wrong_vels.reg'
Region = False

Vsys = 243.  # Koribalski+04
c = 3*10**5  # km/s
Halpha = 6564.61*(Vsys + c)/c
HAlphaLower = optical_vel_to_ang((100-Vsys), Halpha) # Krieger vel model min
HAlphaUpper = optical_vel_to_ang((400-Vsys), Halpha) # Krieger vel model max

# HAlphaLower = Halpha - (Halpha/3000) # velocity widths?
# HAlphaUpper =  Halpha + (Halpha/3000) # widths?

# HAlphaLower = Halpha - 5 # velocity widths?
# HAlphaUpper =  Halpha + 5 # widths?

print_every_other = True
run_parabolas = False

### ---- END FUNCTIONS AND VARIABLES ---- ###

### --- FIT PARABOLAS TO HALPHA TO GET VELOCITIES --- ###
if run_parabolas == True:
    # create the cube centered on HAlpha
    cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                       ContLower2, ContUpper2, Region=Region)
    
    ha_cube = cube.spectral_slab(HAlphaLower * u.AA, HAlphaUpper * u.AA)
    
    # grab the x-axis values
    xvals = np.array(ha_cube.spectral_axis)
    
    larger_cube_range = cube.spectral_slab(6540 * u.AA, 6600 * u.AA)
    larger_xvals_range = np.array(larger_cube_range.spectral_axis)
    
    # loop over the cube
    count = 0
    pixcount = 0
    z, y, x = ha_cube.shape
    parcube = np.zeros((2,y,x)) # cube to store the wavelengths and velocities
    for i in np.arange(y): # y-axis     
        for j in np.arange(x): # x-axis
            yvals = np.array(ha_cube[:,i,j], dtype='float64')
          
            # grab the brightest channel and the two adjacent channels
            
            try:
                ymax = yvals[np.argmax(yvals)]
                yL = yvals[np.argmax(yvals) - 1]
                yR = yvals[np.argmax(yvals) + 1]
                
                # grab the x values corresponding to those channels
                xmax = xvals[np.argmax(yvals)]
                xL = xvals[np.argmax(yvals) - 1]
                xR = xvals[np.argmax(yvals) + 1]
            
                # solve for the coefficients of the parabola
                a,b,c = calc_parabola_vertex(xL, yL, xmax, ymax, xR, yR)
                
                # get the coordinates of the vertex (h, k)
                # h is the x-axis of the vertex and thus what we want for velocity
                h = -b / (2*a);
                k = c - b*b / (4*a)
                
                # convert h (wavelengths) to velocity in km/s
                vel = ang_to_optical_vel(h, Halpha)
                
            except:
                h = np.nan
                vel = np.nan
                
            # make a cube of the wavelengths and velocities
            parcube[:,i,j] = [h, vel]
            
            # update overall counter
            count = count+1
            print('\rCompleted pixel [%s,%s] (%s of %s).' % (j, i, count, (x*y)), end='\r')
            
            # if this is not the next 500th spectrum, update the plot counter
            # and continue
            if print_every_other == True:
                if pixcount % 500 != 0:
                    pixcount = pixcount+1
                    continue
                elif np.isfinite(vel) == False:
                    pixcount = pixcount+1
                    continue
                else:
                    parab = [(a*x**2 + b*x + c) for x in np.linspace(xL, xR, 100)]
                    
                    fig, ax = plt.subplots(constrained_layout=True)
                    plt.step(larger_xvals_range, np.array(larger_cube_range[:,i,j], dtype='float64'), color='tab:gray')
                    plt.plot(np.linspace(xL, xR, 100), parab, color='tab:cyan', label='Parabola',zorder=0)
                    plt.scatter([xL, xmax, xR], [yL, ymax, yR], color='tab:pink', label='Points',zorder=10)
                    plt.scatter(h, k, color='tab:purple', label='Vertex (%s km/s)' % (round(vel,2)),zorder=10)
                    plt.xlabel(r'Wavelength $(\AA)$', fontsize=14)
                    plt.ylabel(r'Flux $(10^{-20} \mathrm{erg/s/cm^2/\AA})$', fontsize=14)
                    
                    # add a secondary axes to show velocity
                    secax = ax.secondary_xaxis('top', functions=(secax_fxn_inverse, secax_fxn))
                    secax.set_xlabel('Velocity (km/s)')
                    ax.axvspan(HAlphaLower, HAlphaUpper, alpha=.1, color='tab:green', label=r'$\lambda_{H \alpha} \pm 5~\AA$')
                    
                    plt.legend(fontsize=11)
                    plt.title('Pixel: %s,%s' % (j,i), fontsize=14)
                    plt.savefig('DiskVelModel/full_cube_KriegerMaxMin/parabolas/pixel_%s_%s.png' % (j, i))
                    plt.close()
                    pixcount = pixcount+1
               
            # or else, print all of the fits
            else:
                if np.isfinite(vel) == False:
                    continue
                parab = [(a*x**2 + b*x + c) for x in np.linspace(xL, xR, 100)]
                
                fig, ax = plt.subplots(constrained_layout=True)
                plt.step(larger_xvals_range, np.array(larger_cube_range[:,i,j], dtype='float64'), color='tab:gray')
                plt.plot(np.linspace(xL, xR, 100), parab, color='tab:cyan', label='Parabola',zorder=0)
                plt.scatter([xL, xmax, xR], [yL, ymax, yR], color='tab:pink', label='Points',zorder=10)
                plt.scatter(h, k, color='tab:purple', label='Vertex (%s km/s)' % (round(vel,2)),zorder=10)
                plt.xlabel(r'Wavelength $(\AA)$', fontsize=14)
                plt.ylabel(r'Flux $(10^{-20} \mathrm{erg/s/cm^2/\AA})$', fontsize=14)
                
                # add a secondary axes to show velocity
                secax = ax.secondary_xaxis('top', functions=(secax_fxn_inverse, secax_fxn))
                secax.set_xlabel('Velocity (km/s)')
                ax.axvspan(HAlphaLower, HAlphaUpper, alpha=.1, color='tab:green', label=r'$\lambda_{H \alpha} \pm 5~\AA$')
                
                plt.legend(fontsize=11)
                plt.title('Pixel: %s,%s' % (j,i), fontsize=14)
                plt.savefig('DiskVelModel/full_cube_KriegerMaxMin/parabolas/pixel_%s_%s.png' % (j, i))
                plt.close()
                pixcount = pixcount+1
           
    w = wcs.WCS(ha_cube[0].header,naxis=2).celestial
    hdr = w.to_header()
    # hdr = fits.Header()
    hdr['FITTYPE'] = 'parabola'
    hdr['PLANE0'] = 'WAVELENGTH'
    hdr['PLANE1'] = 'VELOCITY'
    hdr['UNIT0'] = 'Angstrom'
    hdr['UNIT1'] = 'km/s'
    hdul = fits.PrimaryHDU(data=parcube, header=hdr)
    hdul.writeto('data/ngc253_se_halpha_vel_model_KriegerMaxMin.fits', overwrite=True)
    
    # TIME THE SCRIPT
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
                
                
### --- PROCESS THE HALPHA VELOCITY MAP --- ###


