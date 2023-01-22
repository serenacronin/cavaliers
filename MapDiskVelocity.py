#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:03:27 2022

@author: Serena A. Cronin

This script constructs a velocity map of a galaxy's disk given pixel-by-pixel
spectra in a data cube (i.e., IFU data). If a particular emission line is
expected to be brightest in the disk of the galaxy (rather than, e.g., an
outflow), then one can use these bright peaks to determine a velocity
model of the disk.

This example shows how we can construct a velocity model of the disk of a 
galaxy using Halpha emission, the brightest of which should come from 
the disk.

Smoothing is done using 2D Gaussian kernels, which are necessary for IFU 
data cubes.

Please see https://github.com/serenacronin/SerAstroTools for my 
custom functions used throughout this script.

"""

### ---- MY FUNCTIONS ---- ###
# https://github.com/serenacronin/SerAstroTools
from CubeFitRoutine import CreateCube
from fit_parabola import *
from velocity_conversions import *
from convolution import gauss_2d_kernel

### ---- OTHER FUNCTIONS & PACKAGES ---- ###
import numpy as np
import pyspeckit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from astropy import units as u
from astropy.io import fits
from astropy.wcs import wcs
from astropy.convolution import Gaussian2DKernel, Box2DKernel, convolve, interpolate_replace_nans
import time
import os
startTime = time.time()  # TIME THE SCRIPT


### ---- DEFINE FUNCTIONS AND VARIABLES ---- ###

# functions to get a second axis on the plots
def secax_fxn(xvals):
    
    """
    Get a second axis.
    """
    
    c = 3.0*10**5
    Vsys = 243.
    restwl = 6564.61*(Vsys + c)/c
    return restwl*(xvals + c)/c

def secax_fxn_inverse(xvals):
    
    """
    Get a second axis.
    """
    
    Vsys = 243.
    c = 3.0*10**5
    restwl = 6564.61*(Vsys + c)/c
    return c*((xvals/restwl)-1)

# variables
filename = 'data/ADP.2018-11-22T21_29_46.157.fits'
savepath = 'MapDiskVelocity/'
outfile_dirty = 'ngc253_se_halpha_vel_model_dirty.fits'  # map before cleaning
outfile_method = 'ngc253_se_halpha_vel_model_method.png'  # how the map was cleaned
outfile_smooth = 'ngc253_se_halpha_vel_model_smooth.fits'  # map after cleaning
modelfile = 'data/NGC253_NKrieger_model.fits'  # (optional) model for comparison
model_name = 'Krieger+19'  # (optional) model for comparison

SlabLower = 6500  # focus just on peak (in this case, Halpha)
SlabUpper = 6650
ContUpper1 = 6600  # continuum
ContLower1 = 6545
ContUpper2 = 6750
ContLower2 = 6700
PlotLower = 6540  # range of plot
PlotUpper = 6600
Region = False

Vsys = 243.  # Koribalski+04
c = 3*10**5  # km/s
restwl = 6564.61*(Vsys + c)/c  # Halpha restwl shifted by Vsys of the disk
convention = 'optical'  # velocity convention

# set our peak range; this is a tad arbitrary
peakLower = velocity_to_wavelength(-400, restwl, convention=convention)
peakUpper = velocity_to_wavelength(400, restwl, convention=convention)

beam = 0.8889  # taken from the header
beampix = 4.445  # pixels; one pixel = 0.2arcsec (see ESO archive);
                 # only needed if stamp_out_pix = True
mask_kern = 20*beam  # sigma for 2D Gaussian kernel to do the masking
fill_kern = 40*beam  # sigma for 2D Gaussian kernel to fill in masked values


run_parabolas = False  # trigger parabola fits
print_every_other = True  # print every other parabola fit
n = 100  # number of parabolas we want printed if print_every_other = True
no_print = False  # don't print ANY parabola fit
clean_map = True  # clean up the velocity map
stamp_out_pix = False  # get rid of the pixels in the islands of nans
stampkern = beampix  # only needed if you want to stamp out islands of nans
add_comparison_model = True


## --- FIT PARABOLAS TO AN EMISSION LINE PEAK TO GET VELOCITIES --- ###
if run_parabolas == True:
    
    # create the cube centered on the emission peak you want to work with
    cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                        ContLower2, ContUpper2, Region=Region)
    
    # zero-in on that peak to avoid any other emission lines
    peak_cube = cube.spectral_slab(peakLower * u.AA, peakUpper * u.AA)
    
    # grab the x-axis values
    xvals = np.array(peak_cube.spectral_axis)
    
    # choosing a larger range for plotting purposes
    larger_cube_range = cube.spectral_slab(PlotLower * u.AA, PlotUpper * u.AA)
    larger_xvals_range = np.array(larger_cube_range.spectral_axis)
    
    # loop over the cube
    count = 0
    pixcount = 0
    z, y, x = peak_cube.shape
    parcube = np.zeros((2,y,x)) # cube to store the wavelengths and velocities
    for i in np.arange(y): # y-axis     
        for j in np.arange(x): # x-axis
            yvals = np.array(peak_cube[:,i,j], dtype='float64')
          
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
                a, b, c = calc_parabola_coeffs(xL, yL, xmax, ymax, xR, yR)
 
                # get the coordinates of the vertex (h, k)
                # h is the x-axis of the vertex and thus what we want for velocity
                h, k = calc_parabola_vertex(a, b, c)

                # convert h (wavelengths) to velocity in km/s
                vel = wavelength_to_velocity(h, restwl, convention=convention)
                
            except:
                h = np.nan
                vel = np.nan
                
            # make a cube of the wavelengths and velocities
            parcube[:,i,j] = [h, vel]
            
            # update overall counter
            count = count+1
            print('\rCompleted pixel [%s,%s] (%s of %s).' % (j, i, count, (x*y)), end='\r')
            
            # if this is not the next nth spectrum, update the plot counter
            # and continue
            if print_every_other == True:
                
                # make a directory to store the parabolas
                if not os.path.exists('%sparabolas/' % savepath):
                    os.makedirs('%sparabolas/' % savepath)
                
                # do every n pixels
                if pixcount % n != 0:
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
                    ax.axvspan(peakLower, peakUpper, alpha=.1, color='tab:green', label='velocity range')
                    
                    plt.legend(fontsize=11)
                    plt.title('Pixel: %s,%s' % (j,i), fontsize=14)
                    plt.savefig('%sparabolas/pixel_%s_%s.png' % (savepath, j, i))
                    plt.close()
                    pixcount = pixcount+1
               
            # or print all of the fits
            elif no_print == False:
                if np.isfinite(vel) == False:
                    continue
                
                # make a directory to store the parabolas
                if not os.path.exists('%sparabolas/' % savepath):
                    os.makedirs('%sparabolas/' % savepath)
                    
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
                ax.axvspan(peakLower, peakUpper, alpha=.1, color='tab:green', label='velocity range')
                
                plt.legend(fontsize=11)
                plt.title('Pixel: %s,%s' % (j,i), fontsize=14)
                plt.savefig('%sparabolas/pixel_%s_%s.png' % (savepath, j, i))
                plt.close()
                pixcount = pixcount+1
                
            # or print no fits!
            else:
                pass
           
    w = wcs.WCS(peak_cube[0].header,naxis=2).celestial
    hdr = w.to_header()
    hdr['FITTYPE'] = 'parabola'
    hdr['PLANE0'] = 'WAVELENGTH'
    hdr['PLANE1'] = 'VELOCITY'
    hdr['UNIT0'] = 'Angstrom'
    hdr['UNIT1'] = 'km/s'
    hdul = fits.PrimaryHDU(data=parcube, header=hdr)
    hdul.writeto('%s%s' % (savepath, outfile_dirty),  overwrite=True)
    
    # TIME THE SCRIPT
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
                
                
### --- PROCESS THE HALPHA VELOCITY MAP --- ###
if clean_map == True:
    
    # Region = 'data/ha_map_wrong_vels.reg'
    hdu0 = fits.open(filename)[1]
    og_data = hdu0.data  # original data
    hdu = fits.open(savepath+outfile_dirty)[0]
    og_peak_map = hdu.data[1] # my Halpha map
    
    fig = plt.figure(figsize=(15,10))
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.5
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.variant"] = "small-caps"
    
    # build the kernel
    kern_arr = gauss_2d_kernel(sigma=mask_kern)
    
    # convolve with the original map; this will find pixels that deviate
    # a lot from their neighbors
    new_image = convolve(og_peak_map, kern_arr, normalize_kernel=False, 
                         nan_treatment='fill')
    
    ax = plt.subplot(2, 3, 1)
    ax.imshow(og_peak_map, vmin=-160, vmax=160, origin='lower', cmap='RdBu_r')
    ax.set_title('Original', fontsize=20)
    ax.tick_params(axis='both', which='both',direction='in',
                   width=2.5, labelsize=16, length=7)
    ax.set_xlabel('x (pixels)', fontsize=20)
    ax.set_ylabel('y (pixels)', fontsize=20)
    
    # set a threshold: we want to find pixels that deviate by +/- 100
    # blank out these pixels; this is our mask!
    mask = np.abs(new_image) < 100
    copy_im = og_peak_map.copy()
    copy_im[~mask] = np.nan
    
    ax = plt.subplot(2, 3, 2)
    im = ax.imshow(mask, origin='lower', cmap='Greys_r')
    ax.set_title('Mask ($\sigma = $ %s)' % round(mask_kern,4), fontsize=20)
    ax.tick_params(axis='both', which='both',direction='in', top=True, right=True,
                   width=2.5, labelsize=16, length=7)
    
    # add an annotation
    at = AnchoredText(
    '%sx beam' % int(round(mask_kern/beam,1)), prop=dict(size=18), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    ax = plt.subplot(2, 3, 3)
    ax.imshow(copy_im, vmin=-160, vmax=160, origin='lower', cmap='RdBu_r')
    ax.set_title('Applied Mask',fontsize=20)
    ax.tick_params(axis='both', which='both',direction='in', top=True, right=True,
                   width=2, labelsize=16, length=7)
    
    # okay, now we have islands of nans...but some still have bad pixels within
    # them! we can go ham at blanking out these pixels by using a box kernel
    # and filling in the isolated pixels w nans
    if stamp_out_pix == True:

        new_kernel = Box2DKernel(stampkern)
        new_image2 = convolve(copy_im, new_kernel, normalize_kernel=False, fill_value = np.nan, nan_treatment='fill')
        
        # new_image2 has worse resolution since we've convolved it again
        # let's just find where new_image2 is nan and blank them in the
        # original image to keep the original resolution
        copy_im[np.isnan(new_image2)] = np.nan
        
        ax = plt.subplot(2, 3, 4)
        ax.imshow(copy_im, vmin=106, vmax=371, origin='lower', cmap='RdBu_r')
        ax.set_title('Box Kernel (w=%s)' % boxkern, fontsize=20)
        ax.tick_params(axis='both', which='both',direction='in', top=True, right=True,
                       width=2, labelsize=16, length=7)
        
    # now we want to fill in these nan pixels with the average
    # of the nearest neighbors
    kernel = Gaussian2DKernel(fill_kern)
    final_image = convolve(copy_im, kernel)
    
    if stamp_out_pix == False:
        ax = plt.subplot(2, 3, 4)
        ax.imshow(final_image, vmin=-160, vmax=160, origin='lower',cmap='RdBu_r')
        ax.set_title('Smoothed Image ($\sigma = $ %s)' % round(fill_kern,4),fontsize=20)
        ax.tick_params(axis='both', which='both',direction='in', top=True, right=True,
                        width=2.5, labelsize=16, length=7)
        
        # add an annotation
        at = AnchoredText(
        '%sx beam' % int(round(fill_kern/beam,1)), prop=dict(size=18), frameon=True, loc='lower right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        
        ax = plt.subplot(2, 3, 5) # set up next subplot
    
    else:
        ax = plt.subplot(2, 3, 4) # set up next subplot
        
    # aaaand lets blank back out those new edges
    ## FIXME: might want og_data[0] instead of og_data[1]
    final_image[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    ax.imshow(final_image, vmin=-160, vmax=160, origin='lower',cmap='RdBu_r')
    ax.set_title('Final Image',fontsize=20)
    ax.tick_params(axis='both', which='both',direction='in', top=True, right=True,
                    width=2.5, labelsize=16, length=7)
    
    # let's add a model to compare
    if add_comparison_model == True:
        hdu_model = fits.open(modelfile)[0]
        model_map = (hdu_model.data) - Vsys # subtract the systemic velocity
        
        ax = plt.subplot(2, 3, 6)
        im = ax.imshow(model_map, vmin=-160, vmax=160, origin='lower',cmap='RdBu_r')
        ax.set_title('%s' % model_name,fontsize=20)
        bar = plt.colorbar(im, fraction=0.046)
        bar.set_label('Velocity (km/s)', fontsize=18)
        bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
        ax.tick_params(axis='both', which='both',direction='in', top=True, right=True,
                       width=2.5, labelsize=16, length=7)
    
    
    plt.tight_layout()
    plt.savefig('%s%s' % (savepath, outfile_method), dpi=200)
    
    # save the final, smoothed image as a fits file
    w = wcs.WCS(hdu.header,naxis=2).celestial
    hdr = w.to_header()
    hdr['FITTYPE'] = 'parabola'
    hdr['PLANE0'] = 'WAVELENGTH'
    hdr['PLANE1'] = 'VELOCITY'
    hdr['UNIT0'] = 'Angstrom'
    hdr['UNIT1'] = 'km/s'
    hdr['MASK_BEAM'] = '%s' % int(round(mask_kern/beam,1))
    hdr['SMOOTH_BEAM'] = '%s' % int(round(fill_kern/beam,1))
    
    hdul = fits.PrimaryHDU(data=final_image, header=hdr)
    hdul.writeto('%s%s' % (savepath, outfile_smooth), overwrite=True)
    