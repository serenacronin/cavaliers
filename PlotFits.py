#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:44:23 2022

@author: serenac
"""
import sys
sys.path.insert(0, '/Users/serenac/Desktop/research/astro_tools/')

import numpy as np
import astropy.io.fits as pyfits
from astropy.io import fits
from astropy import units as u
from spectral_cube import SpectralCube
import regions
import matplotlib.pyplot as plt
import numpy as np
import pyspeckit
import warnings
from matplotlib.lines import Line2D
#from CubeFitRoutine import CreateCube, InputParams
from gauss_tools import one_gaussian, three_gaussian, red_chisq, chisq
import scipy
warnings.filterwarnings("ignore")

def plot_one_fit(xpix, ypix, spec, redchisq, savepath, xmin, xmax, ymax, fluxnorm,
                 input_params, show_components=True):
    
    # set up the plotter
    # spec.plotter.refresh()
    spec.plotter(xmin = xmin, xmax = xmax, ymin = -0.4*ymax)    
    # spec.plotter.refresh()
    spec.measure(fluxnorm = fluxnorm)
    
    
    # Make axes labels
    ## TODO: THIS NEEDS TO BE CLEANED UP AND GENERALIZED
    # if xlabel is 'wavelength_A':
    #     spec.plotter.axis.set_xlabel(r'Wavelength $(\AA)$')
        
    # if ylabel is 'flux':
    #     spec.plotter.axis.set_ylabel(r'Flux $(10^{-20} \mathrm{erg/s/cm^2/\AA})$')
    
    # set axes labels and refresh the plotter
    spec.plotter.axis.set_xlabel(r'Wavelength $(\AA)$')
    spec.plotter.axis.set_ylabel(r'Flux $(10^{-20} \mathrm{erg/s/cm^2/\AA})$')
    # spec.plotter.refresh()
    
    # plot the fit
    spec.specfit.plot_fit(annotate=False, 
                          show_components=False,
                          composite_fit_color='tab:pink',
                          lw=1.5)
    
    # get the fit information to the side of the plot
    spec.specfit.annotate(loc='upper right', labelspacing=0.25, markerscale=0.01, 
                          borderpad=0.1, handlelength=0.1, handletextpad=0.1, 
                          fontsize=10, bbox_to_anchor=(1.7,1))
    
    # spec.plotter.refresh()
    
    # plot the residuals
    spec.specfit.plotresiduals(axis=spec.plotter.axis,
                                clear=False,
                                yoffset=-0.2*ymax, 
                                color='tab:purple',
                                linewidth=1.5)
    
    # plot the components individually if applicable
    if show_components is True:
        spec.specfit.plot_components(component_fit_color='tab:cyan',
                                    lw=1.5)
        custom_lines = [Line2D([0], [0], color='tab:pink', lw=2),
                        Line2D([0], [0], color='tab:cyan', lw=2),
                        Line2D([0], [0], color='tab:purple', lw=2),
                        Line2D([0], [0], color='white', lw=2)]

        plt.legend(custom_lines,['Composite', 'Components', 'Residuals', 
                                  'RedChiSq: %s' % round(redchisq,2)], fontsize=7.5, 
                  loc='upper left')
    else:
        custom_lines = [Line2D([0], [0], color='tab:pink', lw=2),
                        Line2D([0], [0], color='tab:purple', lw=2),
                        Line2D([0], [0], color='white', lw=2)]

        plt.legend(custom_lines,['Fit', 'Residuals', 
                                 'RedChiSq: %s' % round(redchisq,2)], fontsize=7.5, 
                  loc='upper left')


    # plt.annotate('%s' % round(input_params[0],4), xy=(6685, 530), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[1],4), xy=(6685, 490), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[2],4), xy=(6685, 450), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[3],4), xy=(6685, 410), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[4],4), xy=(6685, 370), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[5],4), xy=(6685, 330), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[6],4), xy=(6685, 290), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[7],4), xy=(6685, 250), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[8],4), xy=(6685, 210), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[9],4), xy=(6685, 170), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[10],4), xy=(6685, 130),
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[11],4), xy=(6685, 90), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[12],4), xy=(6685, 50), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[13],4), xy=(6685, 10), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[14],4), xy=(6685, -30), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[15],4), xy=(6685, -70), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[16],4), xy=(6685, -110), 
    #              annotation_clip=False, color='tab:green')
    # plt.annotate('%s' % round(input_params[17],4), xy=(6685, -150), 
    #              annotation_clip=False, color='tab:green')

        
    # make a title and legend
    plt.title('Pixel: %s,%s' % (xpix,ypix))
    
    # adjust the plot so that the annotation can be seen, then save to file
    plt.subplots_adjust(right=0.65)
    plt.xlabel(r'Wavelength $(\AA)$')
    
    plt.savefig('%s/pixel_%s_%s.png' % (savepath, xpix, ypix), 
                dpi=200)
    plt.close() # do not print to the terminal lmao
    return
    

    
def plot_fits(cube, FittingInfo, xmin, xmax, yresid, fluxnorm, 
              xlabel, ylabel, savepath, plot_every = 1, 
              plot_failed_only = False, threshold = False):
    """
    cube:           data cube
    FittingInfo:    all of the parameters used in the fit
    xmin:           lower end of the x-axis range
    xmax:           higher end of the y-axis range
    yresid:         location of the residual plots. Recommended that this is < 0.
    fluxnorm:       flux value for the y-axis to be normalized to.
    xlabel:         wavelength (A)... NEED TO ADD MORE OPTIONS
    ylabel:         flux (erg/s/cm^2/A)... NEED TO ADD MORE OPTIONS
    savepath:       path to save plots
    plot_every:     plot every x number of fits. Default is 1 (i.e, every fit).
    plot_failed_only:  plot only fits that fail (i.e, fits where the reduced
                       chisquare >= some threshold)
    threshold:  user-defined reduced chisquare threshold to determine which
                fits "fail"
    
    """
    
    # initialize a counter that will be used in the nested for loops
    count = 0.
    
    # get the fit info
    guesses = FittingInfo[0]  # initial guesses
    limits = FittingInfo[1]   # what are the limits on the fits?
    limited = FittingInfo[2]  # are these limits True or False?
    tied = FittingInfo[3]     # are any parameters tied to each other?
    
    # get the shape of the cube, then loop over the y and x axes
    z, y, x = cube.shape
    
    for i in np.arange(y): # y-axis
        for j in np.arange(x): # x-axis
            
            # if the this is not the next 1000th spectrum, update the count 
            # and continue
            if count % plot_every != 0:
                count = count+1
                continue
                
            else:
                # extract the spectrum at this location
                spectrum = np.array(cube[:,i,j], dtype='float64')
                
            # if we land on a nan pixel, skip and do not update the count
            if np.isfinite(np.mean(spectrum)) == False:
                continue
            
            else:
                count = count+1
    
            spec = pyspeckit.Spectrum(data=spectrum, 
                                      xarr=np.linspace(min(np.array(cube.spectral_axis)),
                                                       max(np.array(cube.spectral_axis)), 
                                                       len(spectrum)))
            
            spec.plotter(xmin = xmin, xmax = xmax, ymin = yresid)
    
            spec.specfit.multifit(fittype='gaussian',
                                  guesses = guesses, 
                                  limits = limits,
                                  limited = limited,
                                  tied = tied,
                                  annotate = False)
                
            spec.plotter.refresh()
            spec.measure(fluxnorm = fluxnorm)
            
            # let's get the reduced chi-square
            errs = spec.specfit.residuals.std()
            
            amps = []
            cens = []
            sigmas = []
            for line in spec.measurements.lines.keys():
                amps.append(spec.measurements.lines[line]['amp'])
                cens.append(spec.measurements.lines[line]['pos'])
                sigmas.append(spec.measurements.lines[line]['fwhm']/2.355)
                
                
            gauss1 = one_gaussian(np.array(cube.spectral_axis), 
                                  amps[0], cens[0], sigmas[0])
            gauss2 = one_gaussian(np.array(cube.spectral_axis), 
                                  amps[1], cens[1], sigmas[1])
            gauss3 = one_gaussian(np.array(cube.spectral_axis), 
                                  amps[2], cens[2], sigmas[2])
            gauss4 = one_gaussian(np.array(cube.spectral_axis), 
                                  amps[3], cens[3], sigmas[3])
            gauss5 = one_gaussian(np.array(cube.spectral_axis), 
                                  amps[4], cens[4], sigmas[4])
            gauss6 = one_gaussian(np.array(cube.spectral_axis), 
                                  amps[5], cens[5], sigmas[5])
   
            model = gauss1+gauss2+gauss3+gauss4+gauss5+gauss6
            
            redchisq = red_chisq(spectrum, model,
                                  num_params=18, err=errs)
            
            # if we only want to plot the failed fits, then skip if the
            # reduced chisquare is greater than some threshold
            # those less than the threshold are considered "failed"
            if plot_failed_only != False:
                if redchisq > threshold:
                    continue
        
    
            # Make axes labels
            ## THIS NEEDS TO BE CLEANED UP AND GENERALIZED
            if xlabel is 'wavelength_A':
                spec.plotter.axis.set_xlabel(r'Wavelength $(\AA)$')
                
            if ylabel is 'flux':
                spec.plotter.axis.set_ylabel(r'Flux $(10^{-20} \mathrm{erg/s/cm^2/\AA})$')
                
            spec.plotter.refresh()
            spec.specfit.plot_fit(annotate=False, 
                                  show_components=False,
                                  composite_fit_color='tab:pink',
                                  lw=1.5)
            
            spec.specfit.plot_components(component_fit_color='tab:cyan',
                                          lw=1.5)
            
            spec.specfit.plotresiduals(axis=spec.plotter.axis,
                                        clear=False,
                                        yoffset=yresid+50, 
                                        color='tab:purple',
                                        linewidth=1.5)
    
            plt.title('Pixel: %s,%s' % (j,i))
            custom_lines = [Line2D([0], [0], color='tab:pink', lw=2),
                            Line2D([0], [0], color='tab:cyan', lw=2),
                            Line2D([0], [0], color='tab:purple', lw=2),
                            Line2D([0], [0], color='white', lw=2)]
    
            plt.legend(custom_lines,['Composite', 'Components', 'Residuals', 
                                      'RedChiSq: %s' % round(redchisq,2)], fontsize=11, 
                      loc='upper left')
            plt.xlabel('Wavelength ($\AA$)')
            plt.savefig('%s/plot_%s_%s.png' % (savepath, j, i), 
                        dpi=200)
            plt.close()
            
if __name__ == '__main__':
    
    # DEFINE ALL VARIABLES #
    xmin = 6530
    xmax = 6620
    yresid = -150
    fluxnorm = 1e-20
    plot_every = 1000
    #plot_failed_only = True
    #threshold = 5.
    xlabel = 'wavelength_A'
    ylabel = 'flux'
    savepath = '/Users/serenac/Desktop/research/ngc253/test-notebooks/CheckNIIFits/spectrum_plots_nosii/plots5_FailedFits'
    
    R = 3000  # MUSE resolving power
    Vsys = 243.  # Koribalski+04
    c = 3*10**5  # km/s
    z = Vsys / c

    filename = '/Users/serenac/Desktop/research/ngc253/data/ADP.2018-11-22T21_29_46.157.fits'
    SlabLower = 6500
    SlabUpper = 6800
    ContUpper1 = 6600
    ContLower1 = 6545
    ContUpper2 = 6750
    ContLower2 = 6700
    # Region = 'ngc253_se_subcube.reg'

    # air rest wavelengths
    NIIa = 6549.86 # *(1+z)
    Halpha = 6564.61
    NIIb = 6585.27
    # SIIa = 6716.44
    # SIIb = 6730.82
    Voutfl = 50 # an initial guess
    wls = [NIIa*(Voutfl + c)/c, NIIa*(Vsys + c)/c, Halpha*(Voutfl + c)/c, 
           Halpha*(Vsys + c)/c, NIIb*(Voutfl + c)/c, NIIb*(Vsys + c)/c]
    
    # amplitude guesses
    amps = [100, 100, 300, 300, 450, 450]

    # tie the center wavelengths to Halpha
    tie1 = Halpha - NIIa
    tie2 = NIIb - Halpha
    subtied = ['p[7] - %f' % tie1, 'p[10] - %f' % tie1,
               'p[7] + %f' % tie2, 'p[10] + %f' % tie2]

    ties = ['', subtied[0], '', '', subtied[1], '',
            '', '', '', '', '', '',
            '', subtied[2], '', '', subtied[3], '']
    
    
    # EXECUTE THE FUNCTIONS #
    
    # create the cube
    cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                       ContLower2, ContUpper2, Region=False)
    
    # get the initial guesses and other information for the fits
    guesses, limits, limited, tied = InputParams(amps, wls, R, ties)
    FittingInfo = [guesses, limits, limited, tied, (0, 0)]
    
    # run the fits
    plot_fits(cube, FittingInfo, xmin, xmax, yresid, fluxnorm, 
                  xlabel, ylabel, savepath,
                  plot_every)