#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:16:22 2022

@author: serenac
"""

import numpy as np
from astropy import units as u
from astropy.io import fits
from spectral_cube import SpectralCube
import pyspeckit
from multiprocessing import Pool
from functools import partial
import regions
from gauss_tools import one_gaussian, red_chisq
from PlotFits import plot_one_fit
import os
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
                nested_fit_threshold=False,
                nested_amps=False, nested_wls=False, nested_ties=False,
                failed_fit_threshold=False, savepath=False, 
                save_good_fits=False, continuum_limits=False):
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
            
            tie1 = Halpha_Wvl - NIIa_Wvl
            tie2 = NIIb_Wvl - Halpha_Wvl
            
            subtied = ['p[4] - %f' % tie1, 'p[4] - %f' % tie1,
                       'p[4] + %f' % tie2, 'p[4] + %f' % tie2]

            ties = ['', subtied[0], '', '', subtied[1], '',
                    '', '', '', '', '', '',
                    '', subtied[2], '', '', subtied[3], '']
    
    nested_fit_threshold : int or float; default=False
        Reduced chi-square value that will trigger a new fit. For example,
        if you have an emission line that sometimes splits into two components
        (via galactic winds, e.g.), then you can first try to fit one Gaussian
        to the line. If the reduced chi square of this fit is less than some
        threshold value, indicating a bad fit, two Gaussians can then be 
        attempted to fit to the line.
        
    nested_amps : list of ints or floats; default=False
        Amplitude guesses of the new fit if a nested fit is triggered.
        
    nested_wls : list of ints or floats; default=False
        Center wavelength guesses of the new fit if a nested fit is triggered.
        
    nested_ties : list of strings; default=False
        Tie parameters together for the new fit if a nested fit is triggered.
        
    failed_fit_threshold : int or float; default=False
        Reduced chi-square value that will trigger a printout of a spectrum
        and its failed fit. Will result in a .png file of each pixel that has
        a failed fit.
        
    savepath : string; default=False
        Location to save the failed fits.
        
    save_good_fits : bool; default=False
        Option to save good fits too. Requires failed_fit_threshold value.
        
    continuum_limits : list; default=False
        If a reduced chi square is calculated, we need to compute the rms. These
        are the lower and upper limits of the good channels that are blanked
        before taking the root mean square of the leftover continuum.
        Format is, e.g., [5000, 6000].

    """
    if nested_fit_threshold != False and type(nested_fit_threshold) is not float:
        raise ValueError('Reduced chi-square value must be False or float.')
    elif nested_fit_threshold != False and nested_amps == False:
        raise ValueError('Need the amplitude parameters of the nested fit.')
    elif nested_fit_threshold != False and nested_wls == False:
        raise ValueError('Need the wavelength parameters of the nested fit.')

    if len(amps) != len(wls):
        raise ValueError('Amps and wavelengths must have the same length.')
    if type(nested_fit_threshold) is float and (len(nested_amps) != len(nested_wls)):
        raise ValueError('Nested amps and wavelengths must have the same length.')
        
    if (save_good_fits is not False) and (failed_fit_threshold is False):
        raise ValueError('Need a failed fit threshold to dictate which fits are "good."')
        
    if (nested_fit_threshold is not False) and (continuum_limits is False):
        raise ValueError('To compute the errors for a chi square, we need lower and upper limits of good channels.')
    elif (failed_fit_threshold is not False) and (continuum_limits is False):
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
    if type(nested_fit_threshold) is float:
        
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
               nested_fit_threshold, failed_fit_threshold, 
               savepath, save_good_fits, continuum_limits,
               nested_guesses, nested_limits, nested_limited, nested_tied])
        
    else:
        return([guesses, limits, limited, point_start, tied, 
               nested_fit_threshold, failed_fit_threshold, savepath, save_good_fits])
    
    
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
            FittingInfo[5] : nested_fit_threshold
            FittingInfo[6] : failed_fit_threshold
            FittingInfo[7] : savepath
            FittingInfo[8] : save_good_fits
            FittingInfo[9] : [lower and upper limits to blank emission lines] (optional)
            FittingInfo[10] : nested_guesses (optional)
            FittingInfo[11] : nested_limits (optional)
            FittingInfo[12] : nested_limited (optional)
            FittingInfo[13] : nested_tied (optional)
            
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
    
    # if (len(FittingInfo) == 8.) is False:
    #     raise ValueError('''Missing one of the parameters for fitting.
    #                       need guesses, limits, limited, tied,
    #                       starting point, a nested fit threshold (or False),
    #                       a failed fit threshold (or False), 
    #                       and a savepath (or False).''')

    guesses = FittingInfo[0]  # initial guesses
    limits = FittingInfo[1]   # what are the limits on the fits?
    limited = FittingInfo[2]  # are these limits True or False?
    point_start = FittingInfo[3]  # which point do we need to start from?
    tied = FittingInfo[4]    # are any parameters tied to each other?
    nested_fit_threshold = FittingInfo[5] # do we want to trigger a new fit
                                          # given some reduced chi square?
    failed_fit_threshold = FittingInfo[6] # do we want to trigger a printout
                                          # of failed fits given a redchisq?
    savepath = FittingInfo[7]             # savepath for the failed fits
    save_good_fits = FittingInfo[8]       # do we want to save the good fits too?
    continuum_limits = FittingInfo[9]   # to calculate the chi square, we need 
                                        # to compute the rms of each spectrum

    chunk_num = chunk_list[0]  # chunk number
    chunk = chunk_list[1]      # chunk of the cube

    # fiiiiiiiiiiiiiiiiiiiiit
    mycube = pyspeckit.Cube(cube=chunk)

    # if we do not need a nested fit and we don't care about
    # flagging failed fits right now, then let's just do this simply
    
    if (nested_fit_threshold is False) and (point_start is False):
        raise ValueError('Need a starting point if nested_fit_threshold = False!')
    
    if (failed_fit_threshold is not False) and (savepath is False):
        raise ValueError('Please input a savepath for the failed fits flag.')
    
    
    if (nested_fit_threshold is False) and (failed_fit_threshold is False):
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
        z, y, x = chunk.shape
        ampsmap = np.zeros((z,x,y))
        wlsmap  = np.zeros((z,x,y))
        sigmasmap = np.zeros((z,x,y))
        
        count = 0
        for i in np.arange(y): # y-axis
            for j in np.arange(x): # x-axis
                spectrum = np.array(chunk[:,i,j], dtype='float64')
                minval = min(np.array(chunk.spectral_axis))
                maxval = max(np.array(chunk.spectral_axis))
                x_axis = np.linspace(minval, maxval, len(spectrum))
                spec = pyspeckit.Spectrum(data=spectrum, xarr=x_axis)
                
                spec.specfit.multifit(fittype='gaussian',
                                      guesses = guesses, 
                                      limits = limits,
                                      limited = limited,
                                      tied = tied,
                                      annotate = False)
                
                spec.plotter.refresh()
                spec.measure(fluxnorm = 1e-20)
                
                # calculate the reduced chi square
                # errs = spec.specfit.residuals.std()
                errs = compute_rms(x_axis, spectrum, continuum_limits[0], 
                                   continuum_limits[1])

                # get fit params
                amps = []
                cens = []
                sigmas = []
                for line in spec.measurements.lines.keys():
                    amps.append(spec.measurements.lines[line]['amp']/(1e-20))
                    cens.append(spec.measurements.lines[line]['pos'])
                    sigmas.append(spec.measurements.lines[line]['fwhm']/2.355)
            
                components = [one_gaussian(np.array(chunk.spectral_axis), 
                                      amps[i], cens[i], sigmas[i]) for i in
                           np.arange(len(amps))]
                
                model = sum(components)
                redchisq = red_chisq(spectrum, model,
                                      num_params=len(amps)*3, err=errs)
                
                
                # do we want to redo the fit with new criteria?
                # i.e., do we want a nested fit?
                if nested_fit_threshold is not False:
                    if (redchisq >= 1.3) | (redchisq <= 0.4):
                        
                        nested_guesses = FittingInfo[10]
                        nested_limits = FittingInfo[11]
                        nested_limited = FittingInfo[12]
                        nested_tied = FittingInfo[13]
                        
                        ## FIXME: FITS DON'T SEEM TO BE WORKING PROPERLY
                        #spec.specfit.clear()
                        spec = pyspeckit.Spectrum(data=spectrum, 
                                                  xarr=np.linspace(minval, maxval, 
                                                                   len(spectrum)))
                        spec.specfit.multifit(fittype='gaussian',
                                              guesses = nested_guesses, 
                                              limits = nested_limits,
                                              limited = nested_limited,
                                              tied = nested_tied,
                                              annotate = False)
                        
                        spec.plotter.refresh()
                        spec.measure(fluxnorm = 1e-20)
                        
                        # errs = spec.specfit.residuals.std()
                        errs = compute_rms(x_axis, spectrum, continuum_limits[0], 
                                           continuum_limits[1])
                        
                        # save the fit params
                        amps = []
                        cens = []
                        sigmas = []
                        for line in spec.measurements.lines.keys():
                            amps.append(spec.measurements.lines[line]['amp']/(1e-20))
                            cens.append(spec.measurements.lines[line]['pos'])
                            sigmas.append(spec.measurements.lines[line]['fwhm']/2.355)
                            
                        # re-calculate the reduced chi-square
                        components = [one_gaussian(np.array(chunk.spectral_axis), 
                                              amps[i], cens[i], sigmas[i]) for i in
                                   np.arange(len(amps))]
                        
                        model = sum(components)
                        
                        redchisq = red_chisq(spectrum, model,
                                              num_params=len(amps)*3, err=errs)
                        
                # do we want to flag failed fits?
                # flagging will involve printing out the fit and pixel coord.
                if failed_fit_threshold is not False:
                    if not os.path.exists('%s/failed_fits/' % savepath):
                        os.makedirs('%s/failed_fits/' % savepath)
                    if (redchisq >= 1.3) | (redchisq <= 0.4):
                        ### PRINT OUT THE FIT
                        plot_one_fit(j, i, spec, redchisq, 
                                     savepath = '%s/failed_fits/' % savepath, 
                                     xmin=6530, xmax=6620, 
                                     yresid=-150, fluxnorm=1e-20)
                  
                # do we want to save good fits?
                if save_good_fits is not False:
                    if not os.path.exists('%s/good_fits/' % savepath):
                        os.makedirs('%s/good_fits/' % savepath)
                    if (redchisq <= 1.3) & (redchisq >= 0.4):
                        ### PRINT OUT THE FIT
                        plot_one_fit(j, i, spec, redchisq, 
                                     savepath = '%s/good_fits/' % savepath, 
                                     xmin=6530, xmax=6620, 
                                     yresid=-150, fluxnorm=1e-20)
                        
                ## TODO: GENERALIZE FLUXNORM, XMIN, AND XMAX ABOVE
                        
                ## TODO: ADD A WAY TO SAVE THE FITS ##
                # spec.specfit.savefit()
                ampsmap[:,j,i] == amps
                wlsmap[:,j,i] == cens
                sigmasmap[:,j,i] == sigmas
            
                count = count+1
                print('\rCompleted pixel [%s,%s] (%s of %s).' % (i, j, count, x*y), end='\r')
            
        # combine the fits to one file
        # TODO: MAKE A GOOD HEADER
        hdr = fits.Header()
        hdr['COMMENT'] = "TODO: MAKE A HEADER."
        
        primary_hdu = fits.PrimaryHDU(header=hdr) # this will be just a hdr
        image_hdu = fits.PrimaryHDU(ampsmap)
        image_hdu2 = fits.ImageHDU(wlsmap)
        image_hdu3 = fits.ImageHDU(sigmasmap)
        hdul = fits.HDUList([primary_hdu, image_hdu, image_hdu2, image_hdu3])
        hdul.writeto('fit_%s.fits' % chunk_num)

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
