##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:54:28 2022

@author: Serena A. Cronin

Test run of the full cube on jansky using the Halpha map as an initial guess
and tying the NII amps and widths together! This is the first run of 
parallelization using the Halpha map.

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
from tqdm import tqdm


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
    

def InputParams(amps, wls, R, point_start, 
                ties=False, nested_fit=False, nested_amps=False, 
                nested_wls=False, nested_ties=False, save_failed_fits=False, 
                savepath=False, save_good_fits=False, continuum_limits=False, 
                free_params=False, nearest_neighbor=False,
                random_pix_only=False):
    
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
        two total fits will be compared to determine which is best. In this 
        case, the lowest value is taken as "best."
        
    nested_amps : list of ints or floats; default=False
        Amplitude guesses of the new fit if a nested fit is triggered.
        
    nested_wls : list of ints or floats; default=False
        Center wavelength guesses of the new fit if a nested fit is triggered.
        
    nested_ties : list of strings; default=False
        Tie parameters together for the new fit if a nested fit is triggered.
        
    save_failed_fits : False or int; default=False
        Trigger to printout of a spectrum if the fit has failed. 
        Will result in a .png file of every n pixel that has a failed fit.
        If not False, save_failed_fits = n.
        
    savepath : string; default=False
        Location to save the failed fits.
        
    save_good_fits : False or int; default=False
        Option to save good fits too.
        
    continuum_limits : list; default=False
        If a reduced chi square is calculated, we need to compute the rms. 
        These are the lower and upper limits of the good channels that are 
        blanked before taking the root mean square of the leftover continuum.
        Format is, e.g., [5000, 6000].
        
    free_params : int or list; default=False
        Number of free parameters for each fit in order to calculate the
        reduced chi square. Format is, e.g., 8 or [8, 9].
        
    random_pix_only : False or int; default=False
        Number of random pixels you want to work with.

    """
    
    # some checks!
    if nested_fit != False and nested_fit != True:
        raise ValueError('nested_fit must be True or False.')
    elif nested_fit != False and nested_amps == False:
        raise ValueError('Need the amplitude parameters of the nested fit.')
    elif nested_fit != False and nested_wls == False:
        raise ValueError('Need the wavelength parameters of the nested fit.')
    elif nested_fit != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
    elif save_failed_fits != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
    elif save_good_fits != False and free_params == False:
        raise ValueError('Need the degrees of freedom of the fits.')
        
    if save_failed_fits != False and type(save_failed_fits) != int:
        raise ValueError('save_failed_fits must be False or int.')
    if save_good_fits != False and type(save_good_fits) != int:
        raise ValueError('save_good_fits must be False or int.')
        
    if (nested_fit is not False) and (continuum_limits is False):
        raise ValueError('''To compute the errors for a chi square, we need 
                         lower and upper limits of good channels.''')
    elif (save_failed_fits is not False) and (continuum_limits is False):
        raise ValueError('''To compute the errors for a chi square, we need 
                         lower and upper limits of good channels.''')

    # get the lower limit of the sigmas, which is the spectral res.
    # then multiply that by 5 to get the upper limit
    # THEN take the average of the lower limits, multiply by 3
    # to get our initial guess
    # recall: sigma = (wavelength / R) / 2.355 because FWHM = 2.355*sigma
    
    wls_arr = np.array(wls)
    sigmas_arr = (wls_arr/float(R))/2.355
    sigmas_lower_lim = np.median(sigmas_arr[np.isfinite(sigmas_arr)])
    sigmas_upper_lim  = sigmas_lower_lim*5
    sigmas_guess = sigmas_lower_lim*3
    
    # generate guesses: combine the lists one element per list at a time
    guesses = [item for sublist in 
               zip(amps, wls, [sigmas_guess]*len(amps)) for item in sublist]

    # generate the limits for the guesses
    amp_lims = [(0, 0)] * len(amps)
    wl_lims = [(0, 0)] * len(wls)
    sigma_lims = [(sigmas_lower_lim, sigmas_upper_lim)] * len(amps)
    # sigma_lims = [item for item in 
    #               zip(list(sigmas_lower_lim), list(sigmas_upper_lim))]
    limits = [item for sublist in zip(amp_lims, wl_lims, sigma_lims)
              for item in sublist]
    limited = [(True, False), (True, False), (True, True),
               (True, False), (True, False), (True, True),
               (True, False), (True, False), (True, True)]
    
    # set the variable tied to False if ties == False
    if ties is False:
        tied = False
    else:
        tied = ties
        
    # if we need a nested fit, then here's what we do
    # same as above, with some extra steps since we're dealing with
    # ndarrays for the wavelengths
    if nested_fit is True:
        
        # given the nature of this calculation, we can use the same
        # sigmas_guess and upper and lower limits as above!
        
        nested_guesses = [item for sublist in 
                          zip(nested_amps, nested_wls, 
                              [sigmas_guess]*len(nested_amps)) 
                          for item in sublist]

        nested_amp_lims = [(0, 0)] * len(nested_amps)
        nested_wl_lims = [(0, 0)] * len(nested_wls)
        # nested_sigma_lims = [item for item in 
        #                      zip(list(n_sigmas_lower_lim), 
        #                          list(n_sigmas_upper_lim))]
        nested_sigma_lims = [(sigmas_lower_lim, sigmas_upper_lim)] * len(nested_amps)
        
        nested_limits = [item for sublist in 
                         zip(nested_amp_lims, nested_wl_lims, 
                             nested_sigma_lims)
                         for item in sublist]
        
        nested_limited = [(True, False), (True, False), (True, True),
                          (True, False), (True, False), (True, True),
                          (True, False), (True, False), (True, True),
                          (True, False), (True, False), (True, True),
                          (True, False), (True, False), (True, True),
                          (True, False), (True, False), (True, True)]

        # set the variable tied to False if ties == False
        if nested_ties is False:
            nested_tied = False
        else:
            nested_tied = nested_ties
            
        return([guesses, limits, limited, point_start, tied,
               nested_fit, save_failed_fits,
               savepath, save_good_fits, continuum_limits, free_params,
               nested_guesses, nested_limits, nested_limited, nested_tied,
               random_pix_only])
        
    else:
        return([guesses, limits, limited, point_start, tied, 
               nested_fit, save_failed_fits, savepath, save_good_fits, 
               continuum_limits, free_params, random_pix_only])
    

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


def calculate_median_amplitude(cube, chunk_num, multiprocess):
    
    """
    Loop over the entire cube and calculate the median
    amplitude. This will be used in our reduced chi-square 
    calculation, where higher S/N pixels will have different
    reduced chi-square thresholds.
    
    """
    
    z,y,x = cube.shape
    amps = []
    
    if chunk_num == 2 or multiprocess == 1:
        print('Reminder that progress bar is an estimate. It only updates for one chunk!')
        for i in tqdm(np.arange(y), 
                      desc='Calculating median amp. of brightest lines per pixel...',
                      unit='pixel'): # y-axis
            for j in np.arange(x): # x-axis
        
                # get the spectrum and the x-axis
                spectrum = np.array(cube[:,i,j], dtype='float64')
                amps.append(max(spectrum))
                
    else:
        for i in np.arange(y): # y-axis
            for j in np.arange(x): # x-axis
        
                # get the spectrum and the x-axis
                spectrum = np.array(cube[:,i,j], dtype='float64')
                amps.append(max(spectrum))
            
    # take the median of the non-nan pixels
    amps_arr = np.array(amps)
    amps_arr = amps_arr[np.isfinite(amps_arr)]
    return np.median(amps_arr)


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
            FittingInfo[6] : save_failed_fits
            FittingInfo[7] : savepath
            FittingInfo[8] : save_good_fits
            FittingInfo[9] : [lower & upper limits to blank lines] (optional)
            FittingInfo[10] : free_params
            FittingInfo[11] : nested_guesses (optional)
            FittingInfo[12] : nested_limits (optional)
            FittingInfo[13] : nested_limited (optional)
            FittingInfo[14] : nested_tied (optional)
            FittingInfo[15] : random_pix_only (optional)
            
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
    save_failed_fits = FittingInfo[6]    # do we want to trigger a printout
                                        # of failed fits given a redchisq?
    savepath = FittingInfo[7]           # savepath for the good/failed fits
    save_good_fits = FittingInfo[8]     # do we want to save the good fits too?
    continuum_limits = FittingInfo[9]   # to calculate the chi square, we need 
                                        # to compute the rms of each spectrum
    free_params = FittingInfo[10]       # num. of free params for redchisquare
    random_pix_only = FittingInfo[15]   # do we only want to work with a number
                                        # of random pixels?

    chunk_num = chunk_list[0]  # chunk number
    chunk = chunk_list[1]      # chunk of the cube
    chunk_indices = chunk_list[2]  # indices of the chunk wrt the full cube
    multiprocess = chunk_list[3]  # number of processes
    
    # some checks!
    if (nested_fit is False) and (point_start is False):
        raise ValueError('Need a starting point if nested_fit = False!')
    
    if (save_failed_fits is True) and (savepath is False):
        raise ValueError('''Please input a savepath for the 
                         save failed fits flag.''')
    
    if (save_good_fits is True) and (savepath is False):
        raise ValueError('''Please input a savepath for the 
                         save good fits flag.''')
        
    if (nested_fit is True) and (len(free_params) == 1):
        raise ValueError('''Dawg we need degrees of freedom for both fits. 
                         Please write as a list.''')
        
    if (random_pix_only is not False) and (type(random_pix_only) != int):
        raise ValueError('Hmm...random_pix_only must be False or int.')
        
    if save_failed_fits != False:
        if not os.path.exists('%s/failed_fits/' % savepath):
            os.makedirs('%s/failed_fits/' % savepath)
    if save_good_fits != False:
        if not os.path.exists('%s/good_fits/' % savepath):
            os.makedirs('%s/good_fits/' % savepath)

    # time to fit!!!!!!! wooooooooooooooo
    # let's get the cube in a form we can work with! yay pyspeckitttt!!!!!!
    mycube = pyspeckit.Cube(cube=chunk)
    
    # if we are multiprocessing,
    # split up the fitparams based on the indices given
    if multiprocess != 1:
        
        guesses = [guesses[q][chunk_indices[0]:chunk_indices[1], 0:438] 
                      if type(guesses[q]) is np.ndarray 
                      else guesses[q] 
                      for q in range(len(guesses))]

        limits = [limits[q][chunk_indices[0]:chunk_indices[1], 0:438] 
                      if type(limits[q]) is np.ndarray 
                      else limits[q] 
                      for q in range(len(limits))]
        tied = [tied[q][chunk_indices[0]:chunk_indices[1], 0:438] 
                      if type(tied[q]) is np.ndarray 
                      else tied[q] 
                      for q in range(len(tied))]
    
    # if we do not need a nested fit and we don't care about
    # flagging failed fits right now, then let's just do this simply
    if (nested_fit is False) and (save_failed_fits is False) and (save_good_fits is False):
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
        # calculate the median amplitude of the cube; to be used later
        median_amp = calculate_median_amplitude(chunk, chunk_num, multiprocess)
        
        # get guesses and nparams if the fit is nested
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
        
        # create a parameter cube for the fits
        z, y, x = chunk.shape
        parcube = np.zeros((npars,x,y))
        
        # option for only working with a random set of pixels
        # let's get those random pixels!
        if random_pix_only != False:
            mask = np.zeros((x,y))  # mask to store random pixels
            randcount = 0
            np.random.seed(0)  # same random pixels for comparison purposes
            
            while randcount < random_pix_only:
            
                # grab random x and y
                randx = np.random.randint(x)
                randy = np.random.randint(y)
                
                # skip if the pixel is nan
                check_nan = np.array(chunk[:,randy,randx], dtype='float64')
                if np.isfinite(np.mean(check_nan)) == False:
                    if chunk_num == 3 or multiprocess == 1: pbar.update(1)
                    count+=1
                    continue
                
                # change that to a 1
                mask[randx,randy] = 1
                randcount = randcount+1
            
            print('Random pixels chosen!\n')
            
        ### --- LET THE FITTING COMMENCE --- ###
    
        ## TODO: GENERALIZE SO THAT THE CHUNK_NUM WE USE FOR THE
        ## PROGRESS BAR IS THE FATTEST CHUNK (most non-nan pixels)
        if chunk_num == 2 or multiprocess == 1:
            pbar = tqdm(total=x*y, desc='Running fitting routine...')
            
        count = 0
        for i in np.arange(x): # x-axis  
            for j in np.arange(y): # y-axis
        
                # option for only working with a random set of pixels
                if random_pix_only != False: 
                    if mask[i,j] == 0:
                        continue
                    else:
                        spectrum = np.array(chunk[:,j,i], dtype='float64')
                else:
                    spectrum = np.array(chunk[:,j,i], dtype='float64')
                            
                # if we land on a nan pixel, skip
                if np.isfinite(np.mean(spectrum)) == False:
                    if chunk_num == 2: pbar.update(1)
                    count+=1
                    continue
 
                # grab x-axis info and the spectrum
                minval = min(np.array(chunk.spectral_axis))
                maxval = max(np.array(chunk.spectral_axis))
                x_axis = np.linspace(minval, maxval, len(spectrum))
                spec1 = pyspeckit.Spectrum(data=spectrum, xarr=x_axis)
                
                # grab specific pixel value from the array
                total_guesses = [guesses[q][j,i] 
                                 if type(guesses[q]) is np.ndarray 
                                 else guesses[q]
                                 for q in range(len(guesses))]
                
                total_limits = [(limits[q][0][j,i], limits[q][1][j,i]) 
                                if type(limits[q][0]) is np.ndarray 
                                else limits[q] 
                                for q in range(len(limits))]

                # if the model is nan, then we gotta skip the pixel
                if np.isfinite(total_guesses[1]) == False:
                    if chunk_num == 2 or multiprocess == 1: pbar.update(1)
                    count+=1
                    continue
                elif 'nan' in tied[j][i][1]:
                    if chunk_num == 2 or multiprocess == 1: pbar.update(1)
                    count+=1
                    continue
                    
                # perform the fit!
                spec1.specfit.multifit(fittype='gaussian',
                                      guesses = total_guesses, 
                                      limits = total_limits,
                                      limited = limited,
                                      tied = tied[j][i],
                                      annotate = False)
                spec1.measure(fluxnorm = 1e-20)
                
                # get errors for the reduced chi square
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
            
                # calculate the reduced chi square
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
                    
                    ## TODO: make generalized
                    if multiprocess != 1:
                        nested_guesses = [nested_guesses[q][chunk_indices[0]:chunk_indices[1], 0:438] 
                                      if type(nested_guesses[q]) is np.ndarray 
                                      else nested_guesses[q] 
                                      for q in range(len(nested_guesses))]
                        nested_limits = [nested_limits[q][chunk_indices[0]:chunk_indices[1], 0:438] 
                                      if type(nested_limits[q]) is np.ndarray 
                                      else nested_limits[q] 
                                      for q in range(len(nested_limits))]
                        nested_tied = [nested_tied[q][chunk_indices[0]:chunk_indices[1], 0:438] 
                                      if type(nested_tied[q]) is np.ndarray 
                                      else nested_tied[q] 
                                      for q in range(len(nested_tied))]
                    
                    spec2 = pyspeckit.Spectrum(data=spectrum, 
                                              xarr=np.linspace(minval, maxval, 
                                              len(spectrum)))
                    
                    # grab specific pixel value from the array

                    nested_total_guesses = [nested_guesses[q][j,i] 
                                            if type(nested_guesses[q]) is np.ndarray 
                                            else nested_guesses[q] 
                                            for q in range(len(nested_guesses))]
                    
                    nested_total_limits =  [(nested_limits[q][0][j,i], nested_limits[q][1][j,i]) 
                                            if type(nested_limits[q][0]) is np.ndarray 
                                            else nested_limits[q] 
                                            for q in range(len(nested_limits))]
                    
                    # if the model is nan, then we gotta skip the pixel
                    if np.isfinite(nested_total_guesses[1]) == False:
                        if chunk_num == 2 or multiprocess == 1: pbar.update(1)
                        count+=1
                        continue
                    elif 'nan' in nested_tied[j][i][1]:
                        if chunk_num == 2 or multiprocess == 1: pbar.update(1)
                        count+=1
                        continue

                    # perform the fit
                    spec2.specfit.multifit(fittype='gaussian',
                                          guesses = nested_total_guesses, 
                                          limits = nested_total_limits,
                                          limited = nested_limited,
                                          tied = nested_tied[j][i],
                                          annotate = False)
                    spec2.measure(fluxnorm = 1e-20)
                    
                    # get errors for the reduced chi square
                    errs2 = compute_rms(x_axis, spectrum, continuum_limits[0], 
                                        continuum_limits[1])
                    
                    # get the fit params
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
                    
                    
                    # set up the probability to be different depending on 
                    # the S/N of the pixel
                    if max(spectrum) <= median_amp:
                        prob = 0.99999
                    elif (max(spectrum) > median_amp) & (max(spectrum) < 3*median_amp):
                        prob = 0.999999
                    elif (max(spectrum) > 3*median_amp) & (max(spectrum) < 6*median_amp):
                        prob = 0.9999999
                    elif (max(spectrum) > 6*median_amp) & (max(spectrum) < 9*median_amp):
                        prob = 0.99999999
                    elif (max(spectrum) > 9*median_amp) & (max(spectrum) < 12*median_amp):
                        prob = 0.999999999
                    elif max(spectrum) >= 12*median_amp:
                        prob = 0.9999999999

                    # which fit is better?
                    # if redchisq1 is better by over 20%
                    if (redchisq1 < redchisq2) & (abs(1 - redchisq1/redchisq2) > 0.2):
                        redchisq = redchisq1
                        
                        # set up the amps, cens, and sigmas planes
                        # of the parameter cube
                        # i.e., set the second component to nan
                        amps = [amp for sublist in zip([np.nan,np.nan,np.nan], amps1)
                                  for amp in sublist]
                        cens = [cen for sublist in zip([np.nan,np.nan,np.nan], cens1)
                                  for cen in sublist]
                        sigmas = [sigma for sublist in zip([np.nan,np.nan,np.nan], sigmas1)
                                  for sigma in sublist]
                        
                        # reduced chi square upper limit from the chisquare distribution
                        upperlim = sp.stats.chi2.ppf(prob, df=len(spectrum)-free_params1) / (len(spectrum)-free_params1)
                        
                        # will need to redo fit if we want to save fits                        
                        print_guesses = total_guesses
                        print_limits = total_limits
                        print_limited = limited
                        print_tied = tied[j][i]
          
                    else: # this includes if the chi-squares are the same
                        redchisq = redchisq2
                        amps = amps2
                        cens = cens2
                        sigmas = sigmas2
                        
                        # reduced chi square upper limit from the chisquare distribution
                        upperlim = sp.stats.chi2.ppf(prob, df=len(spectrum)-free_params2) / (len(spectrum)-free_params2)
                        
                        # will need to redo fit if we want to save fits
                        print_guesses = nested_total_guesses
                        print_limits = nested_total_limits
                        print_limited = nested_limited
                        print_tied = nested_tied[j][i]
                            
                
                # save everything to the parameter cube!!!!
                params = [par for sublist in zip(amps, cens, sigmas)
                          for par in sublist]
                parcube[:,i,j] = params
                
                # do we want to flag failed fits?
                # flagging will involve printing out the fit and pixel coord.
                if save_failed_fits != False:
                    if (redchisq > upperlim) and (count % save_failed_fits == 0):
                        
                        ## TODO: SEE IF I CAN GET RID OF RE-DOING THE FIT 
                        ## right now this is what I have to do to make the 
                        # annotations right; specfit is getting confused I think
                        spec = pyspeckit.Spectrum(data=spectrum, 
                                                  xarr=np.linspace(minval, maxval, 
                                                                   len(spectrum)))
                        spec.specfit.multifit(fittype='gaussian',
                                              guesses = print_guesses, 
                                              limits =  print_limits,
                                              limited = print_limited,
                                              tied = print_tied,
                                              annotate = False)
                        
                        # print the fit
                        plot_one_fit(i, j, spec, redchisq, 
                                      savepath = '%s/failed_fits/' % savepath, 
                                      xmin=6530, xmax=6620, 
                                      ymax=max(spectrum), fluxnorm=1e-20,
                                      input_params = nested_total_guesses)
                 
                # do we want to save good fits?
                # save_good_fits is an integer that tells you to 
                # save every n good fits
                if save_good_fits != False:
                    if (redchisq <= upperlim) and (count % save_good_fits == 0):
    
                        spec = pyspeckit.Spectrum(data=spectrum, 
                                                  xarr=np.linspace(minval, maxval, 
                                                                   len(spectrum)))
                        spec.specfit.multifit(fittype='gaussian',
                                              guesses = print_guesses, 
                                              limits =  print_limits,
                                              limited = print_limited,
                                              tied = print_tied,
                                              annotate = False)
                        
                        # print the fit
                        plot_one_fit(i, j, spec, redchisq, 
                                      savepath = '%s/good_fits/' % savepath, 
                                      xmin=6530, xmax=6620, 
                                      ymax=max(spectrum), fluxnorm=1e-20,
                                      input_params = nested_total_guesses)
                      
                count += 1
                if chunk_num == 2 or multiprocess == 1: pbar.update(1)
                
                ## TODO: GENERALIZE FLUXNORM, XMIN, AND XMAX ABOVE
                # count+=1
                
                # if chunk_num == 1:
                #     print('\rCompleted pixel %s of %s.' 
                #           % (count, x*y), 
                #           end='\r')
                    
                #     if (count % (x*y)*(0.10)) == 0:


        # make a silly little header
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
            
        # if chunk_num == 1:
        #     print('\nProcess %s complete!' % chunk_num)

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
        chunk_list = [0, cube[:,:,:], (0,437), multiprocess]
        FitRoutine(fitparams, chunk_list) # run the fits!!!!!!!!
        
    # or do we want to parallelize it?
    else:
        
        ### TODO: GENERALIZE THIS FOR ANY NUMBER OF MULTIPROCESSORS ##
        # split up the cube for parallelization!
        chunk_list = [[1, cube[:, 0:109, 0:438], (0,109), multiprocess],
                      [2, cube[:, 109:218, 0:438], (109,218), multiprocess],
                      [3, cube[:, 218:327, 0:438], (218,327), multiprocess],
                      [4, cube[:, 327:437, 0:438]], (327,437), multiprocess]
        
        # chunk_list = [[1,cube[:,0:218,0:438], (0,218), multiprocess],
        #               [2,cube[:,218:437,0:438], (218,437), multiprocess]]
        
        # run the fits!!!!!!!!
        num_processes = multiprocess
        with Pool(num_processes) as p:
            result = list(p.imap(partial(FitRoutine, fitparams), 
                                chunk_list))
            
    return
