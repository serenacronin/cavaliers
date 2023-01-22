#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:10:40 2022

JUST LIKE V5 BUT NOW WE'RE DOING A NEAREST-NEIGHBOR APPROACH FOR THE OUTFLOW.
"""

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import wcs
from spectral_cube import SpectralCube
import pyspeckit
from multiprocessing import Pool
from functools import partial
import regions
from gauss_tools import one_gaussian, red_chisq
from PlotFits import plot_one_fit
import os
import scipy as sp
from reproject import reproject_interp
# import matplotlib.pyplot as plt


def CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
                ContLower2, ContUpper2, Region=False):
    """" 
    This will create a baseline-subtracted data cube centered on the 
    wavelengths we want to look at. Given a region (e.g., using ds9), this can 
    also create a subcube focused on a particular region of a larger data cube.
    
    Parameters
    -------------------
    
    filename : string
        Name of the data cube in .fits format.
        
    SlabLower : int or float
        Minimum value of the spectral axis you want to work with. This will
        create what is known as a spectral slab.
    
    SlabUpper : int or float
        Maximum value of the spectral axis you want to work with. This will
        create what is known as a spectral slab.
        
    ContLower : int or float
        Minimum value of the spectral axis in order to fit the continuum.
        It is important to ensure ContLower and ContUpper deal with line-free
        parts of the entire spectrum.
        
    ContUpper : int or float
        Maxmimum value of the spectral axis in order to fit the continuum.
        It is important to ensure ContLower and ContUpper deal with line-free
        parts of the entire spectrum.
        
    Region : string; default=False
        Filename of the region of the cube you want to work with. Can be taken
        from a ds9 region. This will create a subcube instead of using the
        whole data cube.
    """

    # read in the full data cube
    fullcube = SpectralCube.read(filename, hdu=1).spectral_slab(SlabLower * u.AA, 
                                                                SlabUpper * u.AA)

    if Region is not False:
        # slice into a smaller cube
        region = regions.Regions.read(Region)
        subcube = fullcube.subcube_from_regions(region)
        mycube = subcube

    else:
        mycube = fullcube

    # blank out the emission lines
    spectral_axis = mycube.spectral_axis
    good_channels = ((spectral_axis < ContUpper1*u.AA) |
                     (spectral_axis > ContLower1*u.AA) |
                     (spectral_axis < ContUpper2*u.AA) |
                     (spectral_axis > ContLower2*u.AA))
    masked_cube = mycube.with_mask(good_channels[:, np.newaxis, np.newaxis])

    # take the median of the remaining continuum and subtract the baseline
    med = masked_cube.median(axis=0)
    med_cube = mycube - med
    cube = med_cube.spectral_slab(SlabLower*u.AA, SlabUpper*u.AA)
    return cube
    

def InputParams(amps, wls, R, point_start, ties=False, 
                nested_fit=False,
                nested_amps=False, nested_wls=False, nested_ties=False,
                failed_fit=False, savepath=False, 
                save_good_fits=False, continuum_limits=False, free_params=False,
                nearest_neighbor=False, use_center_model=False):
    """
    This function will compile your input parameters for a Gaussian fit
    to IFS data. At its simplest, it will gather together your amplitude and
    center wavelength guesses. It also requires the resolving power R of the
    instrument and an option to tie the fits of multiple emission lines 
    to each other.
    
    If you want a more complicated nested fit, (e.g., trying to impose
    a double-Gaussian fit to an emission line if a single-Gaussian fails),
    you will need to input a nested fit reduced chi square threshold and 
    additional input parameters. You also have the option of flagging failed 
    fits by inputting another reduced chi square threshold value. This will
    save the fit attempt to a .png file to a savepath.
    
    
    Parameters
    -------------------
    
    amps : list of ints or floats
        Input guesses for the amplitudes.
    
    wls : list of ints or floats
        Input guesses for the wavelengths.
        
    R : int or float
        Resolving power of the instrument.
        
    ties : list of strings; default=False
        Tie parameters together if needed. For example, if you want to tie 
        together the NII doublet to the rest wavelength of HAlpha (which is 
        input parameter 'p[4]'), you can do something like:
            
            tie1 = Halpha_wl - NIIa_wl
            tie2 = NIIb_wl - Halpha_wl
            
            subtied = ['p[4] - %f' % tie1, 'p[4] - %f' % tie1,
                       'p[4] + %f' % tie2, 'p[4] + %f' % tie2]

            ties = ['', subtied[0], '', '', subtied[1], '',
                    '', '', '', '', '', '',
                    '', subtied[2], '', '', subtied[3], '']
    
    nested_fit : bool; default=False
        Optional trigger of another fit. The reduced chi square of the 
        two total fits will be compared to determine which is best. In this case,
        the lowest value is taken as "best."
        
    nested_amps : list of ints or floats; default=False
        Amplitude guesses of the new fit if a nested fit is triggered.
        
    nested_wls : list of ints or floats; default=False
        Center wavelength guesses of the new fit if a nested fit is triggered.
        
    nested_ties : list of strings; default=False
        Tie parameters together for the new fit if a nested fit is triggered.
        
    failed_fit : bool; default=False
        Trigger to printout of a spectrum if the fit has failed. 
        Will result in a .png file of each pixel that has a failed fit.
        
    savepath : string; default=False
        Location to save the failed fits.
        
    save_good_fits : bool; default=False
        Option to save good fits too.
        
    continuum_limits : list; default=False
        If a reduced chi square is calculated, we need to compute the rms. These
        are the lower and upper limits of the good channels that are blanked
        before taking the root mean square of the leftover continuum.
        Format is, e.g., [5000, 6000].
        
    free_params : int or list; default=False
        Number of free parameters for each fit in order to calculate the
        reduced chi square. Format is, e.g., 8 or [8, 9].

    """
    if nested_fit != False and nested_fit != True:
        raise ValueError('nested_fit must be True or False.')
    elif nested_fit != False and nested_amps == False:
        raise ValueError('Need the amplitude parameters of the nested fit.')
    elif nested_fit != False and nested_wls == False:
        raise ValueError('Need the wavelength parameters of the nested fit.')
    elif nested_fit != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
    elif failed_fit != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
    elif save_good_fits != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
        
    if failed_fit != False and failed_fit != True:
        raise ValueError('failed_fit must be True or False.')

    if len(amps) != len(wls):
        raise ValueError('Amps and wavelengths must have the same length.')
    if type(nested_fit) is True and (len(nested_amps) != len(nested_wls)):
        raise ValueError('Nested amps and wavelengths must have the same length.')
        
    if (save_good_fits is True) and (failed_fit is False):
        raise ValueError('Need a failed fit threshold to dictate which fits are "good."')
        
    if (nested_fit is not False) and (continuum_limits is False):
        raise ValueError('To compute the errors for a chi square, we need lower and upper limits of good channels.')
    elif (failed_fit is True) and (continuum_limits is False):
        raise ValueError('To compute the errors for a chi square, we need lower and upper limits of good channels.')

    # generate guesses: combine the lists one element per list at a time
    sigmas = list(np.array(wls)/R)  # sigma = wavelength / R
    guesses = [item for sublist in zip(amps, wls, sigmas) for item in sublist]

    # generate the limits (and limited arg) for the guesses
    amp_lims = [(0, 0)] * len(amps)
    wl_lims = [(0, 0)] * len(wls)
    sigma_lims = [(i, 0) for i in sigmas]
    limits = [item for sublist in zip(amp_lims, wl_lims, sigma_lims)
              for item in sublist]
    limited = [(True, False)]*(len(amps)+len(wls)+len(sigmas))

    # set the variable tied to False if ties == False
    if ties is False:
        tied = False
    else:
        tied = ties
        
    # if we need a nested fit, then here's what we do
    if nested_fit is True:
        
        # generate guesses: combine the lists one element per list at at time
        nested_sigmas = list((np.array(nested_wls)/R)/2)  # sigma = wavelength / R 
                                                        # (divided by 2 because we have 2 components)
        nested_guesses = [item for sublist in zip(nested_amps, nested_wls, nested_sigmas) 
                   for item in sublist]

        # generate the limits (and limited arg) for the guesses
        nested_amp_lims = [(0, 0)] * len(nested_amps)
        nested_wl_lims = [(0, 0)] * len(nested_wls)
        nested_sigma_lims = [(i, 0) for i in nested_sigmas]
        nested_limits = [item for sublist in 
                         zip(nested_amp_lims, nested_wl_lims, nested_sigma_lims)
                         for item in sublist]
        nested_limited = [(True, False)]*(len(nested_amps)+len(nested_wls)+len(nested_sigmas))

        # set the variable tied to False if ties == False
        if nested_ties is False:
            nested_tied = False
        else:
            nested_tied = nested_ties
            
        return([guesses, limits, limited, point_start, tied,
               nested_fit, failed_fit, 
               savepath, save_good_fits, continuum_limits, free_params,
               nested_guesses, nested_limits, nested_limited, nested_tied])
        
    else:
        return([guesses, limits, limited, point_start, tied, 
               nested_fit, failed_fit, savepath, save_good_fits, 
               continuum_limits, free_params])
    
        
def optical_vel_to_ang(vels, restwl):
    
    c = 3.0*10**5
    wls = restwl*(vels + c)/c
    
    return wls


def ModelGuesses(mycube, modelcube, restwl):
    
    # first get the model cube to only have naxis=2
    w = wcs.WCS(mycube[0].header,naxis=2).celestial
    new_header = w.to_header()
    mynewcube = fits.PrimaryHDU(data=mycube[0].data,header=new_header)
    
    w = wcs.WCS(modelcube[0].header,naxis=2).celestial
    new_header = w.to_header()
    newmodelcube = fits.PrimaryHDU(data=modelcube[0].data,header=new_header)

    vels, footprint = reproject_interp(modelcube, mynewcube.header)
    fits.writeto('testing_reproject_subregion.fits', vels, 
                        mynewcube.header, overwrite=True)

    wls = optical_vel_to_ang(vels, restwl)
    
    return wls
    

def InputParamsCenterModel(amps, wls_model, R, point_start, 
                           ties=False, nested_fit=False, nested_amps=False, 
                           nested_wls=False, nested_ties=False, failed_fit=False, 
                           savepath=False, save_good_fits=False, continuum_limits=False, 
                           free_params=False, nearest_neighbor=False):
    
    """
    This function will compile your input parameters for a Gaussian fit
    to IFS data. At its simplest, it will gather together your amplitude and
    center wavelength guesses. It also requires the resolving power R of the
    instrument and an option to tie the fits of multiple emission lines 
    to each other.
    
    If you want a more complicated nested fit, (e.g., trying to impose
    a double-Gaussian fit to an emission line if a single-Gaussian fails),
    you will need to input a nested fit reduced chi square threshold and 
    additional input parameters. You also have the option of flagging failed 
    fits by inputting another reduced chi square threshold value. This will
    save the fit attempt to a .png file to a savepath.
    
    Note that this is similar to InputParams() except that it requires a
    model for the center wavelengths of the disk. This also allows for a
    nearest-neighbor approach to the nested_fit if wanting a double-Gaussian.
    Therefore, the center wavelengths vary on a pixel-by-pixel basis.
    
    Parameters
    -------------------
    
    amps : list of ints or floats
        Input guesses for the amplitudes.
    
    wls : string
        Filename of the wavelength/velocity model of the disk
        
    R : int or float
        Resolving power of the instrument.
        
    ties : list of strings; default=False
        Tie parameters together if needed. For example, if you want to tie 
        together the NII doublet to the rest wavelength of HAlpha (which is 
        input parameter 'p[4]'), you can do something like:
            
            tie1 = Halpha_wl - NIIa_wl
            tie2 = NIIb_wl - Halpha_wl
            
            subtied = ['p[4] - %f' % tie1, 'p[4] - %f' % tie1,
                       'p[4] + %f' % tie2, 'p[4] + %f' % tie2]

            ties = ['', subtied[0], '', '', subtied[1], '',
                    '', '', '', '', '', '',
                    '', subtied[2], '', '', subtied[3], '']
    
    nested_fit : bool; default=False
        Optional trigger of another fit. The reduced chi square of the 
        two total fits will be compared to determine which is best. In this case,
        the lowest value is taken as "best."
        
    nested_amps : list of ints or floats; default=False
        Amplitude guesses of the new fit if a nested fit is triggered.
        
    nested_wls : list of ints or floats; default=False
        Center wavelength guesses of the new fit if a nested fit is triggered.
        
    nested_ties : list of strings; default=False
        Tie parameters together for the new fit if a nested fit is triggered.
        
    failed_fit : bool; default=False
        Trigger to printout of a spectrum if the fit has failed. 
        Will result in a .png file of each pixel that has a failed fit.
        
    savepath : string; default=False
        Location to save the failed fits.
        
    save_good_fits : bool; default=False
        Option to save good fits too.
        
    continuum_limits : list; default=False
        If a reduced chi square is calculated, we need to compute the rms. These
        are the lower and upper limits of the good channels that are blanked
        before taking the root mean square of the leftover continuum.
        Format is, e.g., [5000, 6000].
        
    free_params : int or list; default=False
        Number of free parameters for each fit in order to calculate the
        reduced chi square. Format is, e.g., 8 or [8, 9].

    """
    if nested_fit != False and nested_fit != True:
        raise ValueError('nested_fit must be True or False.')
    elif nested_fit != False and nested_amps == False:
        raise ValueError('Need the amplitude parameters of the nested fit.')
    elif nested_fit != False and nested_wls == False:
        raise ValueError('Need the wavelength parameters of the nested fit.')
    elif nested_fit != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
    elif failed_fit != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
    elif save_good_fits != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
        
    if failed_fit != False and failed_fit != True:
        raise ValueError('failed_fit must be True or False.')
        
    if (save_good_fits is True) and (failed_fit is False):
        raise ValueError('Need a failed fit threshold to dictate which fits are "good."')
        
    if (nested_fit is not False) and (continuum_limits is False):
        raise ValueError('To compute the errors for a chi square, we need lower and upper limits of good channels.')
    elif (failed_fit is True) and (continuum_limits is False):
        raise ValueError('To compute the errors for a chi square, we need lower and upper limits of good channels.')

    use_center_model = True
    sigmas = list(np.array(wls_model)/float(R))  # sigma = wavelength / R
    
    # generate guesses: combine the lists one element per list at a time
    guesses = [item for sublist in zip(amps, wls_model, sigmas) for item in sublist]

    # generate the limits (and limited arg) for the guesses
    amp_lims = [(0, 0)] * len(amps)
    wl_lims = [(0, 0)] * len(wls_model)
    sigma_lims = [(i, 0) for i in sigmas]
    limits = [item for sublist in zip(amp_lims, wl_lims, sigma_lims)
              for item in sublist]
    limited = [(True, False)]*(len(amps)+len(wls_model)+len(sigmas))

    # set the variable tied to False if ties == False
    if ties is False:
        tied = False
    else:
        tied = ties
        
    # if we need a nested fit, then here's what we do
    if nested_fit is True:
        
        # generate guesses: combine the lists one element per list at at time
        nested_sigmas = list((np.array(nested_wls)/float(R))/2.)  # sigma = wavelength / R 
                                                        # (divided by 2 because we have 2 components)
        nested_guesses = [item for sublist in zip(nested_amps, nested_wls, nested_sigmas) 
                   for item in sublist]

        # generate the limits (and limited arg) for the guesses
        nested_amp_lims = [(0, 0)] * len(nested_amps)
        nested_wl_lims = [(0, 0)] * len(nested_wls)
        nested_sigma_lims = [(i, 0) for i in nested_sigmas]
        nested_limits = [item for sublist in 
                         zip(nested_amp_lims, nested_wl_lims, nested_sigma_lims)
                         for item in sublist]
        nested_limited = [(True, False)]*(len(nested_amps)+len(nested_wls)+len(nested_sigmas))

        # set the variable tied to False if ties == False
        if nested_ties is False:
            nested_tied = False
        else:
            nested_tied = nested_ties
            
        return([guesses, limits, limited, point_start, tied,
               nested_fit, failed_fit, 
               savepath, save_good_fits, continuum_limits, free_params,
               nested_guesses, nested_limits, nested_limited, nested_tied,
               use_center_model, nearest_neighbor])
        
    else:
        return([guesses, limits, limited, point_start, tied, 
               nested_fit, failed_fit, savepath, save_good_fits, 
               continuum_limits, free_params, use_center_model,
               nearest_neighbor])
    
def compute_rms(x_axis, spectrum, ContLower, ContUpper):
    
    """
    Take in a spectrum, blank out the emission lines, and calculate the
    root mean square (rms) of the leftover continuum to get the uncertainty
    on the observed values.
    
    """
    
    # blank out the emission lines to get the continuum
    cont_channels = np.where((x_axis > ContUpper) |
                     (x_axis < ContLower))
    
    continuum_vals = spectrum[cont_channels]
    
    # calculate the root mean square of the continuum
    rms = np.sqrt(np.mean(np.square(continuum_vals)))
    
    return rms
    


def FitRoutine(FittingInfo, chunk_list):
    
    """
    This function wraps around the pyspeckit and spectral-cube packages
    to fit Gaussians to emission lines in an IFS data cube. The best way to 
    use this is in conjunction with the InputParams() function, which will 
    generate the initial guesses for the Gaussian models. The function is
    optimized for multiprocessing, but you can also use it on the entire cube.
    For example, you can split the cube in half:
        
        chunk_list = [[1,cube[:,0:218,0:438]],
                      [2,cube[:,218:437,0:438]]]
    
        with Pool(num_processes) as p:
            result = list(p.map(partial(FitRoutine, FittingInfo), 
                                chunk_list))
            
    Or you can just pass in the full cube:
        
        chunk_list = [0, cube]
        FitRoutine(FittingInfo, chunk_list)
        
    See RunFit() if you want the splitting of the cube done for you.
    
    Parameters
    -------------------
    
    FittingInfo : ndarray
        While it is best to generate this via InputParams(), this can also
        be manually done. The following format is required:
    
            FittingInfo[0] : guesses
            FittingInfo[1] : limits
            FittingInfo[2] : limited
            FittingInfo[3] : starting point
            FittingInfo[4] : tied
            FittingInfo[5] : nested_fit
            FittingInfo[6] : failed_fit
            FittingInfo[7] : savepath
            FittingInfo[8] : save_good_fits
            FittingInfo[9] : [lower and upper limits to blank emission lines] (optional)
            FittingInfo[10] : free_params
            FittingInfo[11] : nested_guesses (optional)
            FittingInfo[12] : nested_limits (optional)
            FittingInfo[13] : nested_limited (optional)
            FittingInfo[14] : nested_tied (optional)
            FittingInfo[15] : use_center_model (optional)
            
        See InputParams() for more details.
        
    chunk_list : list
        A multi-dimensional list that splits up the data cube and assigns
        "chunk numbers" to each. The following format is required:
            
            chunk_list = [chunk_number, chunk]
            
        For example, the full cube would be:
            
            chunk_list = [0, cube]
           
        Splitting a (437, 436) cube in half would be:
            
            chunk_list = [[1, cube[:,0:218,0:438]],
                          [2, cube[:,218:437,0:438]]]
            
        This allows for each chunk to be saved in individual fits files, 
        with outputs of 'fit_0.fits' for the full cube and 'fit_1.fits' and
        'fit_2.fits' for splitting the cube in half and running in two
        parallel processes.
        
    """

    guesses = FittingInfo[0]  # initial guesses
    limits = FittingInfo[1]   # what are the limits on the fits?
    limited = FittingInfo[2]  # are these limits True or False?
    point_start = FittingInfo[3]  # which point do we need to start from?
    tied = FittingInfo[4]         # are any parameters tied to each other?
    nested_fit = FittingInfo[5] # do we want to trigger a new fit
                                # given some reduced chi square?
    failed_fit = FittingInfo[6]  # do we want to trigger a printout
                                 # of failed fits given a redchisq?
    savepath = FittingInfo[7]             # savepath for the good/failed fits
    save_good_fits = FittingInfo[8]       # do we want to save the good fits too?
    continuum_limits = FittingInfo[9]   # to calculate the chi square, we need 
                                        # to compute the rms of each spectrum
    free_params = FittingInfo[10]       # number of free params for redchisquare
    use_center_model = FittingInfo[15]  # is there a center model to use?
    nearest_neighbor = FittingInfo[16]

    chunk_num = chunk_list[0]  # chunk number
    chunk = chunk_list[1]      # chunk of the cube

    # fiiiiiiiiiiiiiiiiiiiiit
    mycube = pyspeckit.Cube(cube=chunk)

    # if we do not need a nested fit and we don't care about
    # flagging failed fits right now, then let's just do this simply
    
    if (nested_fit is False) and (point_start is False):
        raise ValueError('Need a starting point if nested_fit = False!')
    
    if (failed_fit is True) and (savepath is False):
        raise ValueError('Please input a savepath for the failed fits flag.')
        
    if (nested_fit is True) and (len(free_params) == 1):
        raise ValueError('Need degrees of freedom for both fits. Please write as a list.')

    
    if (nested_fit is False) and (failed_fit is False):
        if tied is False:
            mycube.fiteach(guesses=guesses,
                           limits=limits,
                           limited=limited,
                           start_from_point=point_start)
            mycube.write_fit('fit_%s.fits' % chunk_num, overwrite=True)
            
        else:
            mycube.fiteach(guesses=guesses,
                           limits=limits,
                           limited=limited,
                           tied=tied,
                           start_from_point=point_start)
            mycube.write_fit('fit_%s.fits' % chunk_num, overwrite=True)
            
    # otherwise, we need to loop over each pixel, compute a reduced
    # chi-square, and then compare it with the threshold values.
    # if the reduced chi square is under the nested fit threshold value, 
    # then we trigger a re-fit using the new parameters set by the nested fit.
    # if we want to flag failed fits and the chi square is under the failed
    # fits value, then we will trigger a printout of the fitted spectrum
    # to be analyzed later.
        
    else:
        
        if nested_fit is True:
            # get the guesses and number of parameters
            nested_guesses = FittingInfo[11]
            npars = max([len(guesses), len(nested_guesses)])
            
            # get the degrees of freedom
            free_params1 = free_params[0]
            free_params2 = free_params[1]
        else:
            npars = len(guesses)
            free_params1 = free_params
            
            
        z, y, x = chunk.shape
        parcube = np.zeros((npars,y,x))
        
        pixcount = 0 # doing every X amount of fits
        count = 0 # progress bar
        for i in np.arange(y): # y-axis     
            for j in np.arange(x): # x-axis
            
                #if the this is not the next 1000th spectrum, update the count 
                #and continue
                if pixcount % 500 != 0:
                    pixcount = pixcount+1
                    continue
                    
                else:
                    # extract the spectrum at this location
                    spectrum = np.array(chunk[:,i,j], dtype='float64')
                    
                # if we land on a nan pixel, skip and do not update the count
                if np.isfinite(np.mean(spectrum)) == False:
                    continue
                
                else:
                    pixcount = pixcount+1
    
                minval = min(np.array(chunk.spectral_axis))
                maxval = max(np.array(chunk.spectral_axis))
                x_axis = np.linspace(minval, maxval, len(spectrum))
                spec1 = pyspeckit.Spectrum(data=spectrum, xarr=x_axis)
                
                if use_center_model == True:
                    # if the guess isn't just a single value and is instead
                    # an array of values specific to each pixel, then
                    # grab that specific pixel value from the array
                    model_guesses = [guesses[q][j,i] if type(guesses[q]) != int else guesses[q] for q in range(len(guesses))]
                    ## FIXME: generalize the limits; aka get rid of the ( , 0)
                    model_limits =  [(limits[q][0][j,i], 0) if type(limits[q][0]) != int else limits[q] for q in range(len(limits))]
                
                    # # replace values where the model is nan
                    # if np.isfinite(model_guesses[1]) is False:
                    #     # TODO: try to get these next 5 lines in a list comprehension
                    #     w = 0
                    #     for g in range(len(model_guesses)):
                    #         if np.isfinite(model_guesses[g]) == False:
                    #             model_guesses[g] = backup_wls[w]
                    #             w = w+1
                        
                    
                    # FIXME: this is temporary just to get the code working
                    # we need to deal with nan pixels in the model

                    if np.isfinite(model_guesses[1]) == False:
                        continue
                    elif 'nan' in tied[j][i][1]:
                        continue
                    
                    spec1.specfit.multifit(fittype='gaussian',
                                          guesses = model_guesses, 
                                          limits = model_limits,
                                          limited = limited,
                                          tied = tied[j][i],
                                          annotate = False)
                    
                else:
                    spec1.specfit.multifit(fittype='gaussian',
                                          guesses = guesses, 
                                          limits = limits,
                                          limited = limited,
                                          # tied = tied,
                                          annotate = False)
                    
                # spec1.plotter.refresh()
                spec1.measure(fluxnorm = 1e-20)
                
                # calculate the reduced chi square
                # errs = spec.specfit.residuals.std()
                errs1 = compute_rms(x_axis, spectrum, continuum_limits[0], 
                                   continuum_limits[1])

                # get fit params
                amps1 = []
                cens1 = []
                sigmas1 = []
                for line in spec1.measurements.lines.keys():
                    amps1.append(spec1.measurements.lines[line]['amp']/(1e-20))
                    cens1.append(spec1.measurements.lines[line]['pos'])
                    sigmas1.append(spec1.measurements.lines[line]['fwhm']/2.355)
            
                components1 = [one_gaussian(np.array(chunk.spectral_axis), 
                                      amps1[i], cens1[i], sigmas1[i]) for i in
                           np.arange(len(amps1))]
                
                model1 = sum(components1)
                redchisq1 = red_chisq(spectrum, model1, num_params=len(amps1)*3, 
                                      err=errs1, free_params=free_params1)
                
                spec1.specfit.clear()
                
                # do we want to redo the fit with new criteria?
                # i.e., do we want a nested fit?
                if nested_fit == True:
                    
                    nested_limits = FittingInfo[12]
                    nested_limited = FittingInfo[13]
                    nested_tied = FittingInfo[14]
                    
                    spec2 = pyspeckit.Spectrum(data=spectrum, 
                                              xarr=np.linspace(minval, maxval, 
                                              len(spectrum)))
                    
                    if use_center_model == True:
                        # if the guess isn't just a single value and is instead
                        # an array of values specific to each pixel, then
                        # grab that specific pixel value from the array
                        nested_model_guesses = [nested_guesses[q][j,i] if type(nested_guesses[q]) is np.ndarray else nested_guesses[q] for q in range(len(nested_guesses))]
                        nested_model_limits =  [(nested_limits[q][0][j,i], 0) if type(nested_limits[q][0]) is np.ndarray else nested_limits[q] for q in range(len(nested_limits))]
                        
                        
                        
                        # FIXME: this is temporary just to get the code working
                        # we need to deal with nan pixels in the model
                        if np.isfinite(nested_model_guesses[1]) == False:
                            continue

                        spec2.specfit.multifit(fittype='gaussian',
                                              guesses = nested_model_guesses, 
                                              limits = nested_model_limits,
                                              limited = nested_limited,
                                              tied = nested_tied[j][i],
                                              annotate = False)
                        
                    else:
                        spec2.specfit.multifit(fittype='gaussian',
                                              guesses = nested_guesses, 
                                              limits = nested_limits,
                                              limited = nested_limited,
                                              tied = nested_tied,
                                              annotate = False)
                            
                    
                    # spec2.plotter.refresh()
                    spec2.measure(fluxnorm = 1e-20)
                    
                    # errs = spec.specfit.residuals.std()
                    errs2 = compute_rms(x_axis, spectrum, continuum_limits[0], 
                                       continuum_limits[1])
                    
                    # save the fit params
                    amps2 = []
                    cens2 = []
                    sigmas2 = []
                    for line in spec2.measurements.lines.keys():
                        amps2.append(spec2.measurements.lines[line]['amp']/(1e-20))
                        cens2.append(spec2.measurements.lines[line]['pos'])
                        sigmas2.append(spec2.measurements.lines[line]['fwhm']/2.355)
                        
                    # re-calculate the reduced chi-square
                    components2 = [one_gaussian(np.array(chunk.spectral_axis), 
                                          amps2[i], cens2[i], sigmas2[i]) for i in
                               np.arange(len(amps2))]
                    
                    model2 = sum(components2)
                    
                    redchisq2 = red_chisq(spectrum, model2, num_params=len(amps2)*3, 
                                          err=errs2, free_params=free_params2)
                    

                    # which fit is better?
                    if redchisq1 < redchisq2:
                        redchisq = redchisq1
                        ## TODO: GENERALIZE THESE NEXT FEW LINES
                        amps = [amp for sublist in zip([np.nan,np.nan,np.nan], amps1)
                                  for amp in sublist]
                        cens = [cen for sublist in zip([np.nan,np.nan,np.nan], cens1)
                                  for cen in sublist]
                        sigmas = [sigma for sublist in zip([np.nan,np.nan,np.nan], sigmas1)
                                  for sigma in sublist]
                        
                        # reduced chi square upper limit from the chisquare distribution
                        upperlim = sp.stats.chi2.ppf(0.99999, df=len(spectrum)-free_params1) / (len(spectrum)-free_params1)
                        #TODO: GENERALIZE THE PROBABILITY
                        
                        ## TODO: SEE IF I CAN GET RID OF RE-DOING THE FIT 
                        ## right now this is what I have to do to make the 
                        # annotations right; specfit is getting confused I think
                        spec = pyspeckit.Spectrum(data=spectrum, 
                                                  xarr=np.linspace(minval, maxval, 
                                                                   len(spectrum)))
                        
                        if use_center_model == True:
                            spec.specfit.multifit(fittype='gaussian',
                                                  guesses = model_guesses, 
                                                  limits =  model_limits,
                                                  limited = limited,
                                                  # tied = tied,
                                                  annotate = False)
                            # print(model_guesses)
                        else:
                            spec.specfit.multifit(fittype='gaussian',
                                                  guesses = guesses, 
                                                  limits = limits,
                                                  limited = limited,
                                                  # tied = tied,
                                                  annotate = False)
                        
                        
                    else: # this includes if the chi-squares are the same
                        redchisq = redchisq2
                        amps = amps2
                        cens = cens2
                        sigmas = sigmas2
                        # reduced chi square upper limit from the chisquare distribution
                        upperlim = sp.stats.chi2.ppf(0.99999, df=len(spectrum)-free_params2) / (len(spectrum)-free_params2)

                        spec = pyspeckit.Spectrum(data=spectrum, 
                                                  xarr=np.linspace(minval, maxval, 
                                                                   len(spectrum)))
                        
                        if use_center_model == True:
                            spec.specfit.multifit(fittype='gaussian',
                                                  guesses = nested_model_guesses, 
                                                  limits =  nested_model_limits,
                                                  limited = limited,
                                                  # tied = tied,
                                                  annotate = False)
                            # print(nested_model_guesses)
                        else:
                            spec.specfit.multifit(fittype='gaussian',
                                                  guesses = nested_guesses, 
                                                  limits = nested_limits,
                                                  limited = nested_limited,
                                                  # tied = nested_tied,
                                                  annotate = False)
     
                        
                # # what is the average residual?
                # errs = np.median(np.abs(spec.specfit.residuals))
                # maxerr = np.max(np.abs(spec.specfit.residuals))
                # print(len(np.where(np.abs(spec.specfit.residuals) > 80.)[0]))
                # # print(errs, maxerr, maxerr/errs)
                
                
                # do we want to flag failed fits?
                # flagging will involve printing out the fit and pixel coord.
                if failed_fit == True:
                    if not os.path.exists('%s/failed_fits/' % savepath):
                        os.makedirs('%s/failed_fits/' % savepath)
                    if redchisq > upperlim:
                        ### PRINT OUT THE FIT
                        plot_one_fit(j, i, spec, redchisq, 
                                      savepath = '%s/failed_fits/' % savepath, 
                                      xmin=6530, xmax=6620, 
                                      yresid=-150, fluxnorm=1e-20,
                                      input_params = nested_model_guesses)
                  
                # do we want to save good fits?
                if save_good_fits == True:
                    if not os.path.exists('%s/good_fits/' % savepath):
                        os.makedirs('%s/good_fits/' % savepath)
                    if redchisq <= upperlim:
                        ### PRINT OUT THE FIT
                        plot_one_fit(j, i, spec, redchisq, 
                                      savepath = '%s/good_fits/' % savepath, 
                                      xmin=6530, xmax=6620, 
                                      yresid=-150, fluxnorm=1e-20,
                                      input_params = nested_model_guesses)
                        
                ## TODO: GENERALIZE FLUXNORM, XMIN, AND XMAX ABOVE
                        
                params = [par for sublist in zip(amps, cens, sigmas)
                          for par in sublist]
                parcube[:,i,j] = params
            
                count = count+1
                print('\rCompleted pixel [%s,%s] (%s of %s).' % (j, i, count, (x*y)/500), end='\r')

        # TODO: MAKE A GOOD HEADER
        hdr = fits.Header()
        hdr['FITTYPE'] = 'gaussian'
        parname = ['Amplitude', 'Center', 'Width'] * int(npars / 3)
        jj = 0
        for ii in range(len(parname)):
            if ii % 3 == 0:
                jj+=1
            kw = "PLANE%i" % ii
            hdr[kw] = parname[ii] + str(jj)
        
        hdul = fits.PrimaryHDU(data=parcube, header=hdr)
        
        try:
            hdul.writeto('%s/fit_%s.fits' % (savepath, chunk_num), overwrite=True)
        except:
            hdul.writeto('fit_%s.fits' % chunk_num, overwrite=True)

    return


def RunFit(cube, fitparams, multiprocess=1):
    
    """"
    Run the fitting process with the option for multiprocessing!
    """
    
    if multiprocess < 1:
        raise ValueError('Multiprocess kwg must be at least 1.')
    elif type(multiprocess) != int:
        raise ValueError('Multiprocess kwg must be an integer.')
        
        
    # do we want the cube fitting process to run like normal?
    if multiprocess == 1:
        chunk_list = [0, cube]
        FitRoutine(fitparams, chunk_list) # run the fits!!!!!!!!
        
    # or do we want to parallelize it?
    else:
        
        ### TODO: GENERALIZE THIS FOR ANY NUMBER OF MULTIPROCESSORS ##
        # split up the cube for parallelization!
        chunk_list = [[1, cube[:, 0:109, 0:438]],
                      [2, cube[:, 109:218, 0:438]],
                      [3, cube[:, 218:327, 0:438]],
                      [4, cube[:, 327:437, 0:438]]]
        
        # chunk_list = [[1,cube[:,0:218,0:438]],
        #               [2,cube[:,218:437,0:438]]]
        
        # run the fits!!!!!!!!
        num_processes = multiprocess
        with Pool(num_processes) as p:
            result = list(p.map(partial(FitRoutine, fitparams), 
                                chunk_list))
            
    return
