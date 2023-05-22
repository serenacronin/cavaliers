#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:16:16 2023

@author: Serena A. Cronin

This script....

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
from PlotFits import plot_one_fit, plot_one_fit_PAPER
import os
import scipy as sp
from reproject import reproject_interp
from tqdm import tqdm
import pandas as pd

def one_gaussian(x_array, amp1, cen1, sigma1):
    return(amp1*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))))



def red_chisq(obs, calc, num_params, err, free_params):
    return((np.sum((obs-calc)**2 / err**2))/(len(obs)-free_params))



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

	if Region != False:
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
	
		
def optical_vel_to_ang(vels, Vsys, restwl):
	
	"""
	Quick function to convert optical velocities
	(in km/s) to wavelengths (in Angstroms).
	
	Parameters
	-------------------
	
	vels : float, int, or array
		One or more velocities in km/s to convert
		to wavelength(s) in Angstroms.
		
	restwl : float or int
		Rest wavelength of the spectral line in Angstroms.
		
	
	"""
	
	c = 3.0*10**5
	wls = restwl*((vels + Vsys) + c)/c
	
	return wls

def slice_from_string(slice_string):
    
	"""
	This function will take in a string and convert it so that
	it can be used to slice numpy arrays.

	Adopted from: 
	https://stackoverflow.com/questions/48494581/converting-an-input-string-to-a-numpy-slice
	"""

	slices = slice_string.split(',')
	if len(slices) > 1:
		sl = [slice_from_string(s.strip()) for s in slices]
	else:
		sl = slice(*[int(x) for x in slice_string.split(':')])
	return sl
	

def ModelGuesses(mycube, modelcube, restwl):
	
	"""
	
	Convert a data cube of velocities (in km/s) to wavelengths
	(in Angstroms) to then be used for initial guesses.
	
	Parameters
	-------------------
	
	mycube : data cube
		One or more velocities in km/s to convert
		to wavelength(s) in Angstroms.
		
	restwl : float or int
		Rest wavelength of the spectral line in Angstroms.
		
	
	"""
	
	# first get the model cube to only have naxis=2
	w = wcs.WCS(mycube[0].header,naxis=2).celestial
	new_header = w.to_header()
	mynewcube = fits.PrimaryHDU(data=mycube[0].data,header=new_header)

	w = wcs.WCS(modelcube[0].header,naxis=2).celestial
	new_header = w.to_header()
	newmodelcube = fits.PrimaryHDU(data=modelcube[0].data,header=new_header)

	vels, footprint = reproject_interp(modelcube, mynewcube.header)

	wls = optical_vel_to_ang(vels, restwl)
	
	return wls



def InputParams(fit1, fit2, fit3, R, free_params, continuum_limits,
				amps1=False, centers1=False, ties1=False, 
				amps2=False, centers2=False, ties2=False,
				amps3=False, centers3=False, ties3=False,
				redchisq_range=False, random_pix_only=False, 
				save_fits=False, savepath=False):
	
	"""
	This function will compile your input parameters for a Gaussian fit
	to IFS data. At its simplest, it will gather together your amplitude and
	center wavelength guesses. It also requires the resolving power R of the
	instrument to calculate the widths and an option to tie the fits of multiple 
	emission lines to each other.

	If you want to compare with more complicated fits (e.g., double Gaussians),
	you may input more fit parameters. The program will re-fit with these new
	parameters and will output all fits performed. You also have the option of 
	flagging failed fits by inputting another reduced chi square threshold value. 
	This will save the fit attempt to a .png file to a savepath.

	# Note that the centers stem from a velocity model of the disk.
	
	Parameters
	-------------------
	
	fit1 : bool
		Toggle True if you want to fit with one Gaussian.

	fit2 : bool
		Toggle True if you want to fit with two Gaussians.

	fit3 : bool
		Toggle True if you want to fit with three Gaussians.

	R : int or float
		Resolving power of the instrument.

	free_params : int or list
		Number of free parameters for each fit in order to calculate the
		reduced chi square. Format is, e.g., 8 or [8, 9].

	continuum_limits : list
		Compute the rms for the reduced chi square.
		These are the lower and upper limits of the good channels that are 
		blanked before taking the root mean square of the leftover continuum.
		Format is, e.g., [5000, 6000].

	amps1 : list of ints or floats; default=False
		Input guesses for the one-Gaussian amplitudes.
	
	centers1 : string; default=False
		Filename of the wavelength/velocity model of the disk.
		
	ties1 : list of strings; default=False
		Tie parameters together if needed. For example, if you want to tie 
		together the NII doublet to the rest wavelength of H-alpha (which is 
		input parameter 'p[4]'), you can do something like:
			
			tie1 = Halpha_wl - NIIa_wl
			tie2 = NIIb_wl - Halpha_wl
			
			subtied = ['p[4] - %f' % tie1, 'p[4] - %f' % tie1,
					   'p[4] + %f' % tie2, 'p[4] + %f' % tie2]

			ties = ['', subtied[0], '', '', subtied[1], '',
					'', '', '', '', '', '',
					'', subtied[2], '', '', subtied[3], '']

	amps2 : list of ints or floats; default=False
		Input guesses for the double-Gaussian amplitudes.
	
	centers2 : string; default=False
		Center wavelength guesses for the double-Gaussian fit.

	ties2 : list of strings; default=False
		See ties_1.

	amps3 : list of ints or floats; default=False
		Input guesses for the triple-Gaussian amplitudes.
	
	centers3 : string; default=False
		Center wavelength guesses for the triple-Gaussian fit.

	ties3 : list of strings; default=False
		See ties_1.

	redchisq_range : False or string; default=False
		Option to focus on a range of the spectrum for the reduced
		chi square calculation.
		
	random_pix_only : False or int; default=False
		Number of random pixels you want to work with.
		
	save_fits : False or int; default=False
		Trigger to printout of a spectrum and its fit. 
		Will result in a .png file of every n pixel.
		If not False, save_failed_fits = n.
		
	savepath : string; default=False
		Location to save the failed fits.

	"""
	
	# some (many) checks!
	if fit1 != False and fit1 != True:
		raise ValueError('fit1 must be True or False.')
	if fit2 != False and fit2 != True:
		raise ValueError('fit2 must be True or False.')
	if fit3 != False and fit3 != True:
		raise ValueError('fit3 must be True or False.')
	if fit1 == True and ((amps1 or centers1 or ties1) == False):
		raise ValueError('If fit1 is True, need amps1, centers1, and ties1.')
	if fit2 == True and ((amps2 or centers2 or ties2) == False):
		raise ValueError('If fit2 is True, need amps2, centers2, and ties2.')
	if fit3 == True and ((amps3 or centers3 or ties3) == False):
		raise ValueError('If fit3 is True, need amps3, centers3, and ties3.')

	if fit1 == True and free_params == False:
		raise ValueError('Need the degrees of freedom of the fits.')
	if fit2 == True and free_params == False:
		raise ValueError('Need the degrees of freedom of the fits.')
	if fit3 == True and free_params == False:
		raise ValueError('Need the degrees of freedom of the fits.')
	if save_fits == True and free_params == False:
		raise ValueError('Need the degrees of freedom of the fits.')
		
	if save_fits != False and type(save_fits) != int:
		raise ValueError('save_fits must be False or int.')
		
	if (fit1 == True) and (continuum_limits == False):
		raise ValueError('''To compute the errors for a chi square, we need 
						 lower and upper limits of good channels.''')
	elif (fit2 == True) and (continuum_limits == False):
		raise ValueError('''To compute the errors for a chi square, we need 
						 lower and upper limits of good channels.''')
	elif (fit3 == True) and (continuum_limits == False):
		raise ValueError('''To compute the errors for a chi square, we need 
						 lower and upper limits of good channels.''')

	if (save_fits == True) and (savepath == False):
		raise ValueError('''Please input a savepath for the 
						 save fits flag.''')
		
	if (fit1 == True) and (fit2 == True) and (fit3 == False) and (len(free_params) != 2):
		raise ValueError('''We need degrees of freedom for two fits!! 
						 Please write as a list.''')
	if (fit2 == True) and (fit2 == False) and (fit3 == True) and (len(free_params) != 2):
		raise ValueError('''We need degrees of freedom for two fits!! 
						 Please write as a list.''')
	if (fit1 == False) and (fit2 == True) and (fit3 == True) and (len(free_params) != 2):
		raise ValueError('''We need degrees of freedom for two fits!! 
						 Please write as a list.''')
	if (fit1 == True) and (fit2 == True) and (fit3 == True) and (len(free_params) != 3):
		raise ValueError('''We need degrees of freedom for three fits!! 
						 Please write as a list.''')
		
	if (random_pix_only != False) and (type(random_pix_only) != int):
		raise ValueError('Hmm...random_pix_only must be False or int.')

	### calculation of the widths: ###
	# get the lower limit of the widths, which is the spectral res.
	# then multiply that by 5 to get the upper limit
	# THEN take the average of the lower limits, multiply by 3
	# to get our initial guess
	# recall: sigma = (wavelength / R) / 2.355 because FWHM = 2.355*sigma
	### 						 	###

	# do we want a fit where there is only one Gaussian component?
	if fit1 == True:
		centers1_arr = np.array(centers1)
		widths1_arr = (centers1_arr/float(R))/2.355
		widths1_lower_lim = np.median(widths1_arr[np.isfinite(widths1_arr)])
		widths1_upper_lim  = widths1_lower_lim*5
		widths1_guess = widths1_lower_lim*3
		
		# generate guesses: combine the lists one element per list at a time
		guesses1 = [item for sublist in 
				zip(amps1, centers1, [widths1_guess]*len(amps1)) for item in sublist]

		# generate the limits for the guesses
		amps1_lims = [(0, 0)] * len(amps1)
		centers1_lims = [(0, 0)] * len(centers1)
		widths1_lims = [(widths1_lower_lim, widths1_upper_lim)] * len(amps1)
		
		limits1 = [item for sublist in zip(amps1_lims, centers1_lims, widths1_lims)
				for item in sublist]
		limited1 = [(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True)]

	# if we don't want 1 fit, set things to False	
	else:
		guesses1 = False
		limits1 = False
		limited1 = False
		ties1 = False
		
	# do we want a fit where there are two Gaussian components?
	# same as above, with some extra steps since we're dealing with
	# ndarrays for the wavelengths
	if fit2 == True:
		
		# have to do some extra finagling since part of the list
		# is just a number, and the other part is a numpy array
		centers2_arr = np.array(centers2)
		widths2_arr = (centers2_arr/float(R))/2.355
		widths2_lower_lim = [np.median(widths2_arr[i][np.isfinite(widths2_arr[i])])
		        			for i in range(len(widths2_arr))
							if type(widths2_arr[i]) == np.ndarray]
		widths2_lower_lim = np.median(widths2_lower_lim)
		widths2_upper_lim  = widths2_lower_lim*5
		widths2_guess = widths2_lower_lim*3

		guesses2 = [item for sublist in zip(amps2, centers2, 
					[widths2_guess]*len(amps2)) for item in sublist]

		amps2_lims = [(0, 0)] * len(amps2)
		centers2_lims = [(0, 0)] * len(centers2)
		widths2_lims = [(widths2_lower_lim, widths2_upper_lim)] * len(amps2)
		
		limits2 = [item for sublist in zip(amps2_lims, centers2_lims, 
					widths2_lims) for item in sublist]
		
		limited2 = [(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True)]

	else:
		guesses2 = False
		limits2 = False
		limited2 = False
		ties2 = False
		
	# option of 3 Gaussian fits
	# same idea as fit2 == True above
	if fit3 == True:
		
		centers3_arr = np.array(centers3)
		widths3_arr = (centers3_arr/float(R))/2.355
		widths3_lower_lim = [np.median(widths3_arr[i][np.isfinite(widths3_arr[i])])
		        			for i in range(len(widths3_arr))
							if type(widths3_arr[i]) == np.ndarray]
		widths3_lower_lim = np.median(widths3_lower_lim)
		widths3_upper_lim  = widths3_lower_lim*5
		widths3_guess = widths3_lower_lim*3

		guesses3 = [item for sublist in zip(amps3, centers3, 
					[widths3_guess]*len(amps3)) for item in sublist]

		amps3_lims = [(0, 0)] * len(amps3)
		centers3_lims = [(0, 0)] * len(centers3)
		widths3_lims = [(widths3_lower_lim, widths3_upper_lim)] * len(amps3)
		
		limits3 = [item for sublist in zip(amps3_lims, centers3_lims, 
					widths3_lims) for item in sublist]
		
		limited3 = [(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True)]
		
	else:
		guesses3 = False
		limits3 = False
		limited3 = False
		ties3 = False

	# return everything we need!
	return([fit1, fit2, fit3, free_params, continuum_limits, 
			guesses1, limits1, limited1, ties1,
			guesses2, limits2, limited2, ties2,
			guesses3, limits3, limited3, ties3,
			redchisq_range, random_pix_only, 
			save_fits, savepath])

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


# def calculate_median_amplitude(cube, chunk_num, multiprocess):
	
# 	"""
# 	Loop over the entire cube and calculate the median
# 	amplitude. This will be used in our reduced chi-square 
# 	calculation, where higher S/N pixels will have different
# 	reduced chi-square thresholds.
	
# 	"""
	
# 	z,y,x = cube.shape
# 	amps = []
	
# 	if chunk_num == 2 or multiprocess == 1:
# 		print('Reminder that progress bar is an estimate. It only updates for one chunk!')
# 		for j in tqdm(np.arange(y), 
# 					  desc='Calculating median amp. of brightest lines per pixel...',
# 					  unit='pixel'): # y-axis
# 			for i in np.arange(x): # x-axis
		
# 				# get the spectrum and the x-axis
# 				spectrum = np.array(cube[:,j,i], dtype='float64')
# 				amps.append(max(spectrum))
				
# 	else:
# 		for j in np.arange(y): # y-axis
# 			for i in np.arange(x): # x-axis
		
# 				# get the spectrum and the x-axis
# 				spectrum = np.array(cube[:,j,i], dtype='float64')
# 				amps.append(max(spectrum))
			
# 	# take the median of the non-nan pixels
# 	amps_arr = np.array(amps)
# 	amps_arr = amps_arr[np.isfinite(amps_arr)]
# 	return np.median(amps_arr)

## the above can be replaced by just taking the median of the cube


def component_order_check(params, fit2 = False, fit3 = False):
	
	
	"""
	
	A very inelegant check to make sure the redshifted and blueshifted
	components are not getting switched. Not sure why the fitter
	wants to do that in some regions (high S/N, where blueshifted 
									  component is zero)
	
	"""
	
	# re-order components for 2 Gaussians
	if fit2 == True:
		if (params[4] < params[1]) | (params[10] < params[7]) | (params[16] < params[13]) | (params[22] < params[19]) |(params[28] < params[25]):
			
			new_params = np.zeros(len(params))

			# first emission line
			a_comp1 = params[0]
			v_comp1 = params[1]
			s_comp1 = params[2]
			
			a_comp2 = params[3]
			v_comp2 = params[4]
			s_comp2 = params[5]
			
			new_params[3], new_params[4], new_params[5] = a_comp1, v_comp1, s_comp1
			new_params[0], new_params[1], new_params[2] = a_comp2, v_comp2, s_comp2
			
			# second emission line
			a_comp1 = params[6]
			v_comp1 = params[7]
			s_comp1 = params[8]
			
			a_comp2 = params[9]
			v_comp2 = params[10]
			s_comp2 = params[11]
			
			new_params[9], new_params[10], new_params[11] = a_comp1, v_comp1, s_comp1
			new_params[6], new_params[7], new_params[8] = a_comp2, v_comp2, s_comp2
			
			# third emission line
			a_comp1 = params[12]
			v_comp1 = params[13]
			s_comp1 = params[14]
			
			a_comp2 = params[15]
			v_comp2 = params[16]
			s_comp2 = params[17]
			
			new_params[15], new_params[16], new_params[17] = a_comp1, v_comp1, s_comp1
			new_params[12], new_params[13], new_params[14] = a_comp2, v_comp2, s_comp2

			# fourth emission line
			a_comp1 = params[18]
			v_comp1 = params[19]
			s_comp1 = params[20]
			
			a_comp2 = params[21]
			v_comp2 = params[22]
			s_comp2 = params[23]
			
			new_params[21], new_params[22], new_params[23] = a_comp2, v_comp2, s_comp2
			new_params[18], new_params[19], new_params[20] = a_comp1, v_comp1, s_comp1

			# fifth emission line
			a_comp1 = params[24]
			v_comp1 = params[25]
			s_comp1 = params[26]
			
			a_comp2 = params[27]
			v_comp2 = params[28]
			s_comp2 = params[29]
			
			new_params[27], new_params[28], new_params[29] = a_comp2, v_comp2, s_comp2
			new_params[24], new_params[25], new_params[26] = a_comp1, v_comp1, s_comp1

		else:
			new_params = params

		# re-order components for 3 Gaussians
		if fit3 == True:
			new_params = np.zeros(len(params))
			if (params[4] < params[1]) | (params[10] < params[7]) | (params[16] < params[13]):
			
				# first emission line
				a_comp1 = params[0]
				v_comp1 = params[1]
				s_comp1 = params[2]
				
				a_comp2 = params[3]
				v_comp2 = params[4]
				s_comp2 = params[5]
				
				new_params[3], new_params[4], new_params[5] = a_comp1, v_comp1, s_comp1
				new_params[0], new_params[1], new_params[2] = a_comp2, v_comp2, s_comp2
				
				# second emission line
				a_comp1 = params[6]
				v_comp1 = params[7]
				s_comp1 = params[8]
				
				a_comp2 = params[9]
				v_comp2 = params[10]
				s_comp2 = params[11]
				
				new_params[9], new_params[10], new_params[11] = a_comp1, v_comp1, s_comp1
				new_params[6], new_params[7], new_params[8] = a_comp2, v_comp2, s_comp2
				
				# third emission line
				a_comp1 = params[12]
				v_comp1 = params[13]
				s_comp1 = params[14]
				
				a_comp2 = params[15]
				v_comp2 = params[16]
				s_comp2 = params[17]
				
				new_params[15], new_params[16], new_params[17] = a_comp1, v_comp1, s_comp1
				new_params[12], new_params[13], new_params[14] = a_comp2, v_comp2, s_comp2

				# fourth emission line
				a_comp1 = params[18]
				v_comp1 = params[19]
				s_comp1 = params[20]
				
				a_comp2 = params[21]
				v_comp2 = params[22]
				s_comp2 = params[23]
				
				new_params[21], new_params[22], new_params[23] = a_comp2, v_comp2, s_comp2
				new_params[18], new_params[19], new_params[20] = a_comp1, v_comp1, s_comp1

				# fifth emission line
				a_comp1 = params[24]
				v_comp1 = params[25]
				s_comp1 = params[26]
				
				a_comp2 = params[27]
				v_comp2 = params[28]
				s_comp2 = params[29]
				
				new_params[27], new_params[28], new_params[29] = a_comp2, v_comp2, s_comp2
				new_params[24], new_params[25], new_params[26] = a_comp1, v_comp1, s_comp1	
			
	return new_params

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
	
			FittingInfo[0] : fit1
			FittingInfo[1] : fit2
			FittingInfo[2] : fit3
			FittingInfo[3] : free_params
			FittingInfo[4] : continuum_limits
			FittingInfo[5] : guesses1
			FittingInfo[6] : limits1
			FittingInfo[7] : limited1
			FittingInfo[8] : ties1
			FittingInfo[9] : guesses2
			FittingInfo[10] : limits2
			FittingInfo[11] : limited2
			FittingInfo[12] : ties2
			FittingInfo[13] : guesses3
			FittingInfo[14] : limits3
			FittingInfo[15] : limited3
			FittingInfo[16] : ties3
			FittingInfo[17] : redchisq_range
			FittingInfo[18] : random_pix_only
			FittingInfo[19] : save_fits
			FittingInfo[20] : savepath
			
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
		with outputs of 'fit_0.fits' for the full cube and 'fit1.fits' and
		'fit2.fits' for splitting the cube in half and running in two
		parallel processes.
		
	"""

	# grab all the input parameters!
	fit1 = FittingInfo[0]
	fit2 = FittingInfo[1]
	fit3 = FittingInfo[2]
	free_params = FittingInfo[3]
	continuum_limits = FittingInfo[4]
	guesses1 = FittingInfo[5]
	limits1 = FittingInfo[6]
	limited1 = FittingInfo[7]
	ties1 = FittingInfo[8]
	guesses2 = FittingInfo[9]
	limits2 = FittingInfo[10]
	limited2 = FittingInfo[11]
	ties2 = FittingInfo[12]
	guesses3 = FittingInfo[13]
	limits3 = FittingInfo[14]
	limited3 = FittingInfo[15]
	ties3 = FittingInfo[16]
	redchisq_range = FittingInfo[17]
	random_pix_only = FittingInfo[18]
	save_fits = FittingInfo[19]
	savepath = FittingInfo[20]

	# grab information for multiprocessing
	chunk_num = chunk_list[0]  # chunk number
	chunk = chunk_list[1]      # chunk of the cube
	chunk_indices = chunk_list[2]  # indices of the chunk wrt the full cube
	multiprocess = chunk_list[3]  # number of processes
	
	# make folders if needed
	# print(os.access(savepath, os.W_OK))
	if (save_fits != False) & (fit1 == True):
		if not os.path.exists('%sfits1/' % savepath):
			os.makedirs('%sfits1/' % savepath)

	if (save_fits != False) & (fit2 == True):
		if not os.path.exists('%sfits2/' % savepath):
			os.makedirs('%sfits2/' % savepath)

	if (save_fits != False) & (fit3 == True):
		if not os.path.exists('%sfits3/' % savepath):
			os.makedirs('%sfits3/' % savepath)


	# make text files to output our parameters
	if fit1 == True:
		# make text files
		# well ok first remove the file if it exists so we can overwrite w new lines
		if os.path.exists("%sfits1.txt" % savepath):
			os.remove("%sfits1.txt" % savepath)

		f1 = open("%sfits1.txt" % savepath, "w")
		f1.write('X,Y,RedChiSq,')
		f1.write('Amp1,Amp2,Amp3,Amp4,Amp5,')
		f1.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,')
		f1.write('Sig1,Sig2,Sig3,Sig4,Sig5\n')

		# save errors
		if os.path.exists("%sfits1_err.txt" % savepath):
			os.remove("%sfits1_err.txt" % savepath)

		e1 = open("%sfits1_err.txt" % savepath, "w")
		e1.write('X,Y,RedChiSq,')
		e1.write('Amp1,Amp2,Amp3,Amp4,Amp5,')
		e1.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,')
		e1.write('Sig1,Sig2,Sig3,Sig4,Sig5\n')

	if fit2 == True:
		# make text files
		# well ok first remove the file if it exists so we can overwrite w new lines
		if os.path.exists("%sfits2.txt" % savepath):
			os.remove("%sfits2.txt" % savepath)

		f2 = open("%sfits2.txt" % savepath, "w")
		f2.write('X,Y,RedChiSq,')
		f2.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,')
		f2.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,')
		f2.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10\n')

		# save errors
		if os.path.exists("%sfits2_err.txt" % savepath):
			os.remove("%sfits2_err.txt" % savepath)

		e2 = open("%sfits2_err.txt" % savepath, "w")
		e2.write('X,Y,RedChiSq,')
		e2.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,')
		e2.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,')
		e2.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10\n')

	if fit3 == True:
		if os.path.exists("%sfits3.txt" % savepath):
			os.remove("%sfits3.txt" % savepath)

		f3 = open("%sfits3.txt" % savepath, "w")
		f3.write('X,Y,RedChiSq,')
		f3.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,Amp11,Amp12,Amp13,Amp14,Amp15')
		f3.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,Wvl11,Wvl12,Wvl13,Wvl14,Wvl15')
		f3.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10,Sig11,Sig12,Sig13,Sig14,Sig15\n')

		# save errors
		if os.path.exists("%sfits3_err.txt" % savepath):
			os.remove("%sfits3_err.txt" % savepath)

		e3 = open("%sfits3_err.txt" % savepath, "w")
		e3.write('X,Y,RedChiSq,')
		e3.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,Amp11,Amp12,Amp13,Amp14,Amp15')
		e3.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,Wvl11,Wvl12,Wvl13,Wvl14,Wvl15')
		e3.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10,Sig11,Sig12,Sig13,Sig14,Sig15\n')

	# get the cube in a form we can work with
	# for the fitting using pyspeckit
	# mycube = pyspeckit.Cube(cube=chunk)
	
	# if we are multiprocessing,
	# split up the fitparams based on the indices given
	if multiprocess != 1:
		
		# TODO: generalize 0:438
		guesses1 = [guesses1[q][chunk_indices[0]:chunk_indices[1], 0:438]
					  if type(guesses1[q]) == np.ndarray 
					  else guesses1[q] 
					  for q in range(len(guesses1))]

		limits1 = [limits1[q][chunk_indices[0]:chunk_indices[1], 0:438] 
					  if type(limits1[q]) == np.ndarray 
					  else limits1[q] 
					  for q in range(len(limits1))]
		ties1 = [ties1[q][chunk_indices[0]:chunk_indices[1], 0:438] 
					  if type(ties1[q]) == np.ndarray 
					  else ties1[q] 
					  for q in range(len(ties1))]
			
	# we need to loop over each pixel, calculate
	# each fit, and output the parameters to a text file

	# get the total number of parameters
	# and number of degrees of freedom (i.e., free params)
	if (fit1 == True) & (fit2 == False) & (fit3 == False):  # if we only have fit1
		free_params1 = free_params[0]
	elif (fit1 == False) & (fit2 == True) & (fit3 == False):  # if we only have fit2
		free_params2 = free_params[0]
	elif (fit1 == False) & (fit2 == False) & (fit3 == True):  # if we only have fit3
		free_params3 = free_params[0]
	elif (fit1 == True) & (fit2 == True) & (fit3 == False):  # if we have both fit1 and fit2
		free_params1 = free_params[0]
		free_params2 = free_params[1]
	elif (fit1 == True) & (fit2 == False) & (fit3 == True):  # if we have both fit1 and fit3
		free_params1 = free_params[0]
		free_params3 = free_params[1]
	elif (fit1 == False) & (fit2 == True) & (fit3 == True):  # if we have both fit2 and fit3
		free_params2 = free_params[0]
		free_params3 = free_params[1]
	elif (fit1 == True) & (fit2 == True) & (fit3 == True):  # if we have all three!
		free_params1 = free_params[0]
		free_params2 = free_params[1]
		free_params3 = free_params[2]
	
	# create two parameter cubes for the fits
	# z-dimension is npars + 1 because we want an
	# extra parameter to save the reduced chi squares
	_, y, x = chunk.shape

	# option for only working with a random set of pixels
	# let's get those random pixels!
	if random_pix_only != False:
		mask = np.zeros((y,x))  # mask to store random pixels
		randcount = 0
		np.random.seed(0)  # same random pixels for comparison purposes
		
		while randcount < random_pix_only:
		
			# grab random x and y
			randx = np.random.randint(x)
			randy = np.random.randint(y)
			
			# skip if the pixel is nan
			check_nan = np.array(chunk[:,randy,randx], dtype='float64')
			if np.isfinite(np.mean(check_nan)) == False:
				continue
			
			# change that to a 1
			mask[randy,randx] = 1
			randcount = randcount+1
		
		print('%s random pixels chosen!\n' % randcount)
		
	### --- LET THE FITTING COMMENCE --- ###

	## TODO: GENERALIZE SO THAT THE CHUNK_NUM WE USE FOR THE
	## PROGRESS BAR IS THE FATTEST CHUNK (most non-nan pixels)
	if chunk_num == 2 or multiprocess == 1:
		pbar = tqdm(total=x*y, desc='Running fitting routine...')
		
	count = 0

	for i in np.arange(x): # x-axis 

		if (i != 249):
			continue

		for j in np.arange(y): # y-axis

			if (j != 148):
				continue
	
			# option for only working with a random set of pixels
			if random_pix_only != False:
				if mask[j,i] == 0:
					if chunk_num == 2 or multiprocess == 1: pbar.update(1)  
					continue
				else:
					spectrum = np.array(chunk[:,j,i], dtype='float64')
			else:
				spectrum = np.array(chunk[:,j,i], dtype='float64')
						
			# if we land on a nan pixel, skip
			if np.isfinite(np.mean(spectrum)) == False:
				if chunk_num == 2 or multiprocess == 1: pbar.update(1)  
				count+=1
				continue

			# grab x-axis info and the spectrum
			minval = min(np.array(chunk.spectral_axis))
			maxval = max(np.array(chunk.spectral_axis))
			x_axis = np.linspace(minval, maxval, len(spectrum))
			
			
#################################################################################################################### 
# ONE COMPONENT FIT
####################################################################################################################
			if fit1 == True:
				# grab the spectrum
				spec1 = pyspeckit.Spectrum(data=spectrum, xarr=x_axis)

				# grab specific pixel value from the array
				total_guesses1 = [guesses1[q][j,i] 
									if type(guesses1[q]) == np.ndarray 
									else guesses1[q]
									for q in range(len(guesses1))]
				
				total_limits1 = [(limits1[q][0][j,i], limits1[q][1][j,i]) 
								if type(limits1[q][0]) == np.ndarray 
								else limits1[q] 
								for q in range(len(limits1))]

				# if the model is nan, then we gotta skip the pixel
				if np.isfinite(total_guesses1[1]) == False:
					if chunk_num == 2 or multiprocess == 1: pbar.update(1)
					count+=1
					continue
					
				# perform the fit!
				spec1.specfit.multifit(fittype='gaussian',
										guesses = total_guesses1, 
										limits = total_limits1,
										limited = limited1,
										tied = ties1,
										annotate = False)
				spec1.measure(fluxnorm = 1e-20) # TODO: GENERALIZE FLUXNORM
				
				# get errors on the spectrum for the reduced chi square
				errs1 = compute_rms(x_axis, spectrum, continuum_limits[0], 
									continuum_limits[1])

				# get fit params
				amps1_list = []
				centers1_list = []
				widths1_list = []
				for line in spec1.measurements.lines.keys():
					amps1_list.append(round(spec1.measurements.lines[line]['amp']/(1e-20), 4)) #TODO: GENERALIZE
					centers1_list.append(round(spec1.measurements.lines[line]['pos'], 4))
					widths1_list.append(round(spec1.measurements.lines[line]['fwhm']/2.355,4))

				# grab the error on each parameter
				err_params1 = [round(e,4) for e in spec1.specfit.parinfo.errors]

				# save the parameters
				params1 = [par for sublist in zip(amps1_list, centers1_list, widths1_list)
							for par in sublist]
			
				# calculate the reduced chi square
				components1 = [one_gaussian(np.array(chunk.spectral_axis), 
								amps1_list[i], centers1_list[i], widths1_list[i]) for i in
								np.arange(len(amps1_list))]
				model1 = sum(components1)

				# what we want is to do the reduced chi square only over the range of emission lines
				# bc we have such a long baseline between emission lines
				# FIXME: generalize this				
				chans_ind = np.argwhere(((np.array(chunk.spectral_axis) > 6525) & (np.array(chunk.spectral_axis) < 6620)) |
						((np.array(chunk.spectral_axis) > 6700) & (np.array(chunk.spectral_axis) < 6750)))
				
				# grab the channels we want from the spectrum itself and our model
				chans_spec = np.array(spectrum)[chans_ind]
				chans_model1 = model1[chans_ind]

				redchisq1 = round(red_chisq(chans_spec, chans_model1, 
					num_params=len(amps1_list)*3, err=errs1, 
					free_params=free_params1),4)
				
				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plot_one_fit(i, j, spec1, redchisq1, 
									savepath = '%sfits1' % savepath, 
									xmin=6500, xmax=6800, 
									ymax=max(spectrum), fluxnorm=1e-20,
									input_params = total_guesses1)
						
				# save parameters to file
				with open("%sfits1.txt" % savepath, "a") as f1:
					f1.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s\n' %
							(i, j, redchisq1,
							params1[0], params1[3], params1[6], params1[9], params1[12],
	     					params1[1], params1[4], params1[7], params1[10], params1[13],
						    params1[2], params1[5], params1[8], params1[11], params1[14]))
				f1.close()

				# save errors on parameters to file
				with open("%sfits1_err.txt" % savepath, "a") as e1:
					e1.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s\n' %
							(i, j, redchisq1,
							err_params1[0], err_params1[3], err_params1[6], err_params1[9], err_params1[12],
	     					err_params1[1], err_params1[4], err_params1[7], err_params1[10], err_params1[13],
						    err_params1[2], err_params1[5], err_params1[8], err_params1[11], err_params1[14]))
				e1.close()


#################################################################################################################### 
# TWO COMPONENT FIT
####################################################################################################################
			
			# do we want to to fit with 2 Gaussians?
			if fit2 == True:
				
				## TODO: make generalized
				# if we are multiprocessing, make sure we are
				# working with the correct chunk of guesses
				
				# grab specific pixel value from the array
				total_guesses2 = [guesses2[q][j,i] 
									if type(guesses2[q]) == np.ndarray 
									else guesses2[q] 
									for q in range(len(guesses2))]
			
				
				total_limits2 =  [(limits2[q][0][j,i], limits2[q][1][j,i]) 
										if type(limits2[q][0]) == np.ndarray 
										else limits2[q] 
										for q in range(len(limits2))]
				
				# if the model is nan, then we gotta skip the pixel
				if np.isfinite(total_guesses2[1]) == False:
					if chunk_num == 2 or multiprocess == 1: pbar.update(1)
					count+=1
					continue

				# grab the spectrum
				spec2 = pyspeckit.Spectrum(data=spectrum, xarr=np.linspace(minval, maxval, 
											len(spectrum)))

				# perform the fit
				spec2.specfit.multifit(fittype='gaussian',
										guesses = total_guesses2, 
										limits = total_limits2,
										limited = limited2,
										# tied = ties2[j][i],
										tied = ties2,
										annotate = False)
				spec2.measure(fluxnorm = 1e-20) # TODO: generalize
				
				# get errors for the reduced chi square
				errs2 = compute_rms(x_axis, spectrum, continuum_limits[0], 
									continuum_limits[1])
				
				# get the fit params
				amps2_list = []
				centers2_list = []
				widths2_list = []
				for line in spec2.measurements.lines.keys():
					amps2_list.append(round(spec2.measurements.lines[line]['amp']/(1e-20),4))
					centers2_list.append(round(spec2.measurements.lines[line]['pos'],4))
					widths2_list.append(round(spec2.measurements.lines[line]['fwhm']/2.355,4))

				# grab the error on each parameter
				err_params2 = [round(e,4) for e in spec2.specfit.parinfo.errors]

				# save all of the parameters
				params2 = [par for sublist in zip(amps2_list, centers2_list, widths2_list)
							for par in sublist]

				# calculate reduced chi-square; first add up each Gaussian
				components2 = [one_gaussian(np.array(chunk.spectral_axis), 
								amps2_list[i], centers2_list[i], widths2_list[i]) 
								for i in np.arange(len(amps2_list))]
				model2 = sum(components2)
				
				# what we want is to do the reduced chi square only over the range of emission lines
				# bc we have such a long baseline between emission lines
				# FIXME: generalize this		
				chans_ind = np.argwhere(((np.array(chunk.spectral_axis) > 6525) & (np.array(chunk.spectral_axis) < 6620)) |
						((np.array(chunk.spectral_axis) > 6700) & (np.array(chunk.spectral_axis) < 6750)))
				
				# grab the channels we want from the spectrum itself and our model
				chans_spec = np.array(spectrum)[chans_ind]
				chans_model2 = model2[chans_ind]

				redchisq2 = round(red_chisq(chans_spec, chans_model2, 
					num_params=len(amps2_list)*3, err=errs2, 
					free_params=free_params2),4)

				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plot_one_fit_PAPER(i, j, spec2, redchisq2, 
									savepath = '%sfits2' % savepath, 
									xmin=6500, xmax=6800,
									ymax=max(spectrum), fluxnorm=1e-20,
									input_params = total_guesses2)

				# save parameters to file
				with open("%sfits2.txt" % savepath, "a") as f2:
					f2.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' %
							(i, j, redchisq2,
							params2[0], params2[3], params2[6], params2[9], params2[12],
	     					params2[15], params2[18], params2[21], params2[24], params2[27],
						    params2[1], params2[4], params2[7], params2[10], params2[13],
							params2[16], params2[19], params2[22], params2[25], params2[28],
	     					params2[2], params2[5], params2[8], params2[11], params2[14],
						    params2[17], params2[20], params2[23], params2[26], params2[29]))
				f2.close()


				# save errors on parameters to file
				with open("%sfits2_err.txt" % savepath, "a") as e2:
					e2.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' %
							(i, j, redchisq2,
							err_params2[0], err_params2[3], err_params2[6], err_params2[9], err_params2[12],
	     					err_params2[15], err_params2[18], err_params2[21], err_params2[24], err_params2[27],
						    err_params2[1], err_params2[4], err_params2[7], err_params2[10], err_params2[13],
							err_params2[16], err_params2[19], err_params2[22], err_params2[25], err_params2[28],
	     					err_params2[2], err_params2[5], err_params2[8], err_params2[11], err_params2[14],
						    err_params2[17], err_params2[20], err_params2[23], err_params2[26], err_params2[29]))
				e2.close()
						
#################################################################################################################### 
# THREE COMPONENT FIT
####################################################################################################################
			
			# do we want to to fit with 3 Gaussians?
			if fit3 == True:
				
				## TODO: make generalized
				# if we are multiprocessing, make sure we are
				# working with the correct chunk of guesses
				if multiprocess != 1:
					guesses3 = [guesses3[q][chunk_indices[0]:chunk_indices[1], 0:438] 
									if type(guesses3[q]) == np.ndarray 
									else guesses3[q] 
									for q in range(len(guesses3))]
					limits3 = [limits3[q][chunk_indices[0]:chunk_indices[1], 0:438] 
									if type(limits3[q]) == np.ndarray 
									else limits3[q] 
									for q in range(len(limits3))]
					ties3 = [ties3[q][chunk_indices[0]:chunk_indices[1], 0:438] 
									if type(ties3[q]) == np.ndarray 
									else ties3[q] 
									for q in range(len(ties3))]
				
				# grab specific pixel value from the array
				total_guesses3 = [guesses3[q][j,i] 
									if type(guesses3[q]) == np.ndarray 
									else guesses3[q] 
									for q in range(len(guesses3))]
				
				total_limits3 =  [(limits3[q][0][j,i], limits3[q][1][j,i]) 
										if type(limits3[q][0]) == np.ndarray 
										else limits3[q] 
										for q in range(len(limits3))]
				
				# if the model is nan, then we gotta skip the pixel
				if np.isfinite(total_guesses3[1]) == False:
					if chunk_num == 2 or multiprocess == 1: pbar.update(1)
					count+=1
					continue

				# grab the spectrum
				spec3 = pyspeckit.Spectrum(data=spectrum, xarr=np.linspace(minval, maxval, 
											len(spectrum)))

				# perform the fit
				spec3.specfit.multifit(fittype='gaussian',
										guesses = total_guesses3, 
										limits = total_limits3,
										limited = limited3,
										# tied = ties2[j][i],
										tied = ties3,
										annotate = False)
				spec3.measure(fluxnorm = 1e-20) # TODO: generalize
				
				# get errors for the reduced chi square
				errs3 = compute_rms(x_axis, spectrum, continuum_limits[0], 
									continuum_limits[1])
				
				# get the fit params
				amps3_list = []
				centers3_list = []
				widths3_list = []
				for line in spec3.measurements.lines.keys():
					amps3_list.append(round(spec3.measurements.lines[line]['amp']/(1e-20),4))
					centers3_list.append(round(spec3.measurements.lines[line]['pos'],4))
					widths3_list.append(round(spec3.measurements.lines[line]['fwhm']/2.355,4))

				# grab the error on each parameter
				err_params3 = [round(e,4) for e in spec3.specfit.parinfo.errors]

				# save the parameters
				params3 = [par for sublist in zip(amps3_list, centers3_list, widths3_list)
							for par in sublist]

				# calculate reduced chi-square; first add up each Gaussian
				components3 = [one_gaussian(np.array(chunk.spectral_axis), 
								amps3_list[i], centers3_list[i], widths3_list[i]) 
								for i in np.arange(len(amps3_list))]
				model3 = sum(components3)
				
				# what we want is to do the reduced chi square only over the range of emission lines
				# bc we have such a long baseline between emission lines
				# FIXME: generalize this		
				chans_ind = np.argwhere(((np.array(chunk.spectral_axis) > 6525) & (np.array(chunk.spectral_axis) < 6620)) |
						((np.array(chunk.spectral_axis) > 6700) & (np.array(chunk.spectral_axis) < 6750)))
				
				# grab the channels we want from the spectrum itself and our model
				chans_spec = np.array(spectrum)[chans_ind]
				chans_model3 = model3[chans_ind]

				redchisq3 = round(red_chisq(chans_spec, chans_model3, 
					num_params=len(amps3_list)*3, err=errs3, 
					free_params=free_params3),4)
				
				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plot_one_fit(i, j, spec3, redchisq3, 
									savepath = '%s/fits3' % savepath, 
									xmin=6500, xmax=6800,
									ymax=max(spectrum), fluxnorm=1e-20,
									input_params = total_guesses3)
						
				# save parameters to file
				with open("%sfits3.txt" % savepath, "a") as f3:
					f3.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
							 '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
							'%s, %s, %s\n' %
							(i, j, redchisq3,
							params3[0], params3[3], params3[6], params3[9], params3[12],
	     					params3[15], params3[18], params3[21], params3[24], params3[27],
						    params3[30], params3[33], params3[36], params3[39], params3[42],
							params3[1], params3[4], params3[7], params3[10], params3[13],
	     					params3[16], params3[19], params3[22], params3[25], params3[28],
						    params3[31], params3[34], params3[37], params3[40], params3[43],
							params3[2], params3[5], params3[8], params3[11], params3[14],
							params3[17], params3[20], params3[23], params3[26], params3[29],
							params3[32], params3[35], params3[38], params3[41], params3[44]))
				f3.close()

				# save errors on parameters to file
				with open("%sfits3_err.txt" % savepath, "a") as e3:
					e3.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
	      					'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
							 '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
							'%s, %s, %s\n' %
							(i, j, redchisq3,
							err_params3[0], err_params3[3], err_params3[6], err_params3[9], err_params3[12],
	     					err_params3[15], err_params3[18], err_params3[21], err_params3[24], err_params3[27],
						    err_params3[30], err_params3[33], err_params3[36], err_params3[39], err_params3[42],
							err_params3[1], err_params3[4], err_params3[7], err_params3[10], err_params3[13],
	     					err_params3[16], err_params3[19], err_params3[22], err_params3[25], err_params3[28],
						    err_params3[31], err_params3[34], err_params3[37], err_params3[40], err_params3[43],
							err_params3[2], err_params3[5], err_params3[8], err_params3[11], err_params3[14],
							err_params3[17], err_params3[20], err_params3[23], err_params3[26], err_params3[29],
							err_params3[32], err_params3[35], err_params3[38], err_params3[41], err_params3[44]))
				e3.close()
				
			# up the counter + progress bar
			count += 1
			if chunk_num == 2 or multiprocess == 1: pbar.update(1)
			## TODO: GENERALIZE FLUXNORM, XMIN, AND XMAX ABOVE

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
	# TODO: GENERALIZE

	if multiprocess == 1:
		chunk_list = [0, cube[:,:,:], (0,437), multiprocess]
		# chunk_list = [0, cube[:,210:250,273:304], (0,437), multiprocess]
		FitRoutine(fitparams, chunk_list)
		
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
		
		
		# run the fits
		num_processes = multiprocess
		with Pool(num_processes) as p:
			result = list(p.imap(partial(FitRoutine, fitparams), 
								chunk_list))
			
	return