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
from gauss_tools import one_gaussian, red_chisq
from PlotFits import plot_one_fit
import os
import scipy as sp
from reproject import reproject_interp
from tqdm import tqdm
import pickle


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
	wls = restwl*(vels + c)/c
	
	return wls
	

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
	# fits.writeto('testing_reproject_subregion.fits', vels, 
	#                     mynewcube.header, overwrite=True)

	wls = optical_vel_to_ang(vels, restwl)
	
	return wls



def InputParams(fit1, fit2, R, free_params, continuum_limits,
				amps1=False, centers1=False, ties1=False, 
				amps2=False, centers2=False, ties2=False,
				random_pix_only=False, save_fits=False, savepath=False):
	
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

	Note that the centers stem from a velocity model of the disk.
	
	Parameters
	-------------------
	
	fit1 : bool
		Toggle True if you want to fit with one Gaussian.

	fit2 : bool
		Toggle True if you want to fit with two Gaussians.

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
	if fit1 == True and ((amps1 or centers1 or ties1) == False):
		raise ValueError('If fit1 is True, need amps_1, centers_1, and ties_1.')
	if fit2 == True and ((amps2 or centers2 or ties2) == False):
		raise ValueError('If fit2 is True, need amps_2, centers_2, and ties_2.')

	if fit1 == True and free_params == False:
		raise ValueError('Need the degrees of freedom of the fits.')
	if fit2 == True and free_params == False:
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

	if (save_fits is True) and (savepath is False):
		raise ValueError('''Please input a savepath for the 
						 save fits flag.''')
		
	if (fit1 is True) and (fit2 is True) and (len(free_params) == 1):
		raise ValueError('''We need degrees of freedom for both fits!! 
						 Please write as a list.''')
		
	if (random_pix_only is not False) and (type(random_pix_only) != int):
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
		# sigma_lims = [item for item in 
		#               zip(list(sigmas_lower_lim), list(sigmas_upper_lim))]
		limits1 = [item for sublist in zip(amps1_lims, centers1_lims, widths1_lims)
				for item in sublist]
		limited1 = [(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True)]
		
	# do we want a fit where there are two Gaussian components?
	# same as above, with some extra steps since we're dealing with
	# ndarrays for the wavelengths
	if fit2 == True:
		
		# given the nature of this calculation, we can use the same
		# widths1_guess and upper and lower limits as above!
		guesses2 = [item for sublist in zip(amps2, centers2, 
					[widths1_guess]*len(amps2)) for item in sublist]

		amps2_lims = [(0, 0)] * len(amps2)
		centers2_lims = [(0, 0)] * len(centers2)
		widths2_lims = [(widths1_lower_lim, widths1_upper_lim)] * len(amps2)
		
		limits2 = [item for sublist in zip(amps2_lims, centers2_lims, 
					widths2_lims) for item in sublist]
		
		limited2 = [(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True),
					(True, False), (True, False), (True, True)]

	# return everything we need!
	return([fit1, fit2, free_params, continuum_limits, 
			guesses1, limits1, limited1, ties1,
			guesses2, limits2, limited2, ties2,
			random_pix_only, save_fits, savepath])

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
		for j in tqdm(np.arange(y), 
					  desc='Calculating median amp. of brightest lines per pixel...',
					  unit='pixel'): # y-axis
			for i in np.arange(x): # x-axis
		
				# get the spectrum and the x-axis
				spectrum = np.array(cube[:,j,i], dtype='float64')
				amps.append(max(spectrum))
				
	else:
		for j in np.arange(y): # y-axis
			for i in np.arange(x): # x-axis
		
				# get the spectrum and the x-axis
				spectrum = np.array(cube[:,j,i], dtype='float64')
				amps.append(max(spectrum))
			
	# take the median of the non-nan pixels
	amps_arr = np.array(amps)
	amps_arr = amps_arr[np.isfinite(amps_arr)]
	return np.median(amps_arr)


def component_order_check(params):
	
	
	"""
	
	A very inelegant check to make sure the redshifted and blueshifted
	components are not getting switched. Not sure why the fitter
	wants to do that in some regions (high S/N, where blueshifted 
									  component is zero)
	
	"""
	
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
		
		new_params[12], new_params[13], new_params[14] = a_comp1, v_comp1, s_comp1
		new_params[15], new_params[16], new_params[17] = a_comp2, v_comp2, s_comp2
		
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
			FittingInfo[2] : free_params
			FittingInfo[3] : continuum_limits
			FittingInfo[4] : guesses1
			FittingInfo[5] : limits1
			FittingInfo[6] : limited1
			FittingInfo[7] : ties1
			FittingInfo[8] : guesses2
			FittingInfo[9] : limits2
			FittingInfo[10] : limited2
			FittingInfo[11] : ties2
			FittingInfo[12] : random_pix_only
			FittingInfo[13] : save_fits
			FittingInfo[14] : savepath
			
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
	free_params = FittingInfo[2]
	continuum_limits = FittingInfo[3]
	guesses1 = FittingInfo[4]
	limits1 = FittingInfo[5]
	limited1 = FittingInfo[6]
	ties1 = FittingInfo[7]
	guesses2 = FittingInfo[8]
	limits2 = FittingInfo[9]
	limited2 = FittingInfo[10]
	ties2 = FittingInfo[11]
	random_pix_only = FittingInfo[12]
	save_fits = FittingInfo[13]
	savepath = FittingInfo[14]

	# grab information for multiprocessing
	chunk_num = chunk_list[0]  # chunk number
	chunk = chunk_list[1]      # chunk of the cube
	chunk_indices = chunk_list[2]  # indices of the chunk wrt the full cube
	multiprocess = chunk_list[3]  # number of processes
	
	# make folders if needed
	if (save_fits != False) & (fit1 == True):
		if not os.path.exists('%s/fits1/' % savepath):
			os.makedirs('%s/fits1/' % savepath)

	if (save_fits != False) & (fit2 == True):
		if not os.path.exists('%s/fits2/' % savepath):
			os.makedirs('%s/fits2/' % savepath)

	# get the cube in a form we can work with
	# for the fitting using pyspeckit
	mycube = pyspeckit.Cube(cube=chunk)
	
	# if we are multiprocessing,
	# split up the fitparams based on the indices given
	if multiprocess != 1:
		
		# TODO: generalize 0:438
		guesses1 = [guesses1[q][chunk_indices[0]:chunk_indices[1], 0:438]
					  if type(guesses1[q]) is np.ndarray 
					  else guesses1[q] 
					  for q in range(len(guesses1))]

		limits1 = [limits1[q][chunk_indices[0]:chunk_indices[1], 0:438] 
					  if type(limits1[q]) is np.ndarray 
					  else limits1[q] 
					  for q in range(len(limits1))]
		ties1 = [ties1[q][chunk_indices[0]:chunk_indices[1], 0:438] 
					  if type(ties1[q]) is np.ndarray 
					  else ties1[q] 
					  for q in range(len(ties1))]
	
	# # if we do not need a nested fit and we don't care about
	# # flagging failed fits right now, then let's just do this simply
	# if (nested_fit is False) and (save_failed_fits is False) and (save_good_fits is False):
	# 	if tied is False:
	# 		mycube.fiteach(guesses=guesses,
	# 					   limits=limits,
	# 					   limited=limited,
	# 					   start_from_point=point_start)
	# 		mycube.write_fit('fit_%s.fits' % chunk_num, overwrite=True)
			
	# 	else:
	# 		mycube.fiteach(guesses=guesses,
	# 					   limits=limits,
	# 					   limited=limited,
	# 					   tied=tied,
	# 					   start_from_point=point_start)
	# 		mycube.write_fit('fit_%s.fits' % chunk_num, overwrite=True)
			
			
	# otherwise, we need to loop over each pixel, calculate
	# a one-component fit and save it to a parameter cube. if we want
	# to flag failed or good fits, we can use a chi-square to trigger
	# a printout of the fitted spectrum. if we want to re-do the fits
	# using one with more parameters, then do so and redo the above.
	# this fit will save to a different cube.
		
	# FIXME: MOVE TO ANALYSIS
	# # calculate the median amplitude of the cube; to be used later
	# # then save as pkl file; if this file already exists
	# # the just open it
	# if not os.path.exists('%s/median_amp.pkl' % savepath):
	# 	median_amp = calculate_median_amplitude(chunk, chunk_num, multiprocess)
	# 	pickle.dump(median_amp, open('%s/median_amp.pkl' % savepath, 'wb'))
	
	# else:
	# 	median_amp = pickle.load(open('%s/median_amp.pkl' % savepath, 'rb'))

	# get the total number of parameters
	# and number of degrees of freedom (i.e., free params)
	if (fit1 == True) & (fit2 == False):
		npars1 = len(guesses1)
		free_params1 = free_params
	elif (fit1 == True) & (fit2 == True):
		npars1 = len(guesses1)
		npars2 = len(guesses2)
		free_params1 = free_params[0]
		free_params2 = free_params[1]
	elif (fit1 == False) & (fit2 == True):
		npars2 = len(guesses2)
		free_params2 = free_params
	
	# create two parameter cubes for the fits
	# z-dimension is npars1 + 1 because we want an
	# extra parameter to save the reduced chi squares
	z, y, x = chunk.shape
	parcube1 = np.zeros((npars1+1, y, x))  # rows, columns
	parcube2 = np.zeros((npars2+1, y, x))  # rows, columns

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
				# if chunk_num == 3 or multiprocess == 1: pbar.update(1)
				# count+=1
				continue
			
			# change that to a 1
			mask[randy,randx] = 1
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
				if mask[j,i] == 0:
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
			
			
			############################# 
			##### ONE COMPONENT FIT #####
			############################# 
			if fit1 == True:
				# grab the spectrum
				spec1 = pyspeckit.Spectrum(data=spectrum, xarr=x_axis)

				# grab specific pixel value from the array
				total_guesses1 = [guesses1[q][j,i] 
									if type(guesses1[q]) is np.ndarray 
									else guesses1[q]
									for q in range(len(guesses1))]
				
				total_limits1 = [(limits1[q][0][j,i], limits1[q][1][j,i]) 
								if type(limits1[q][0]) is np.ndarray 
								else limits1[q] 
								for q in range(len(limits1))]

				# if the model is nan, then we gotta skip the pixel
				if np.isfinite(total_guesses1[1]) == False:
					if chunk_num == 2 or multiprocess == 1: pbar.update(1)
					count+=1
					continue
				elif 'nan' in ties1[j][i][1]:
					if chunk_num == 2 or multiprocess == 1: pbar.update(1)
					count+=1
					continue
					
				# perform the fit!
				spec1.specfit.multifit(fittype='gaussian',
										guesses = total_guesses1, 
										limits = total_limits1,
										limited = limited1,
										tied = ties1[j][i],
										annotate = False)
				spec1.measure(fluxnorm = 1e-20) # TODO: GENERALIZE FLUXNORM
				
				# get errors for the reduced chi square
				errs1 = compute_rms(x_axis, spectrum, continuum_limits[0], 
									continuum_limits[1])

				# get fit params
				amps1_list = []
				centers1_list = []
				widths1_list = []
				for line in spec1.measurements.lines.keys():
					amps1_list.append(spec1.measurements.lines[line]['amp']/(1e-20)) #TODO: GENERALIZE
					centers1_list.append(spec1.measurements.lines[line]['pos'])
					widths1_list.append(spec1.measurements.lines[line]['fwhm']/2.355)
			
				# calculate the reduced chi square
				components1 = [one_gaussian(np.array(chunk.spectral_axis), 
								amps1_list[i], centers1_list[i], widths1_list[i]) for i in
								np.arange(len(amps1_list))]
				model1 = sum(components1)
				redchisq1 = red_chisq(spectrum, model1, num_params=len(amps1_list)*3, 
										err=errs1, free_params=free_params1)
				
				# save everything to the parameter cube!!!!
				params1 = [par for sublist in zip(amps1_list, centers1_list, widths1_list)
							for par in sublist] + redchisq1
				parcube1[:,j,i] = params1
				
				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plot_one_fit(i, j, spec1, redchisq1, 
									savepath = '%s/fits1/' % savepath, 
									xmin=6530, xmax=6620, 
									ymax=max(spectrum), fluxnorm=1e-20,
									input_params = total_guesses1)


			############################# 
			##### TWO COMPONENT FIT #####
			############################# 
			
			# do we want to to fit with 2 Gaussians?
			if fit2 == True:
				
				## TODO: make generalized
				# if we are multiprocessing, make sure we are
				# working with the correct chunk of guesses
				if multiprocess != 1:
					guesses2 = [guesses2[q][chunk_indices[0]:chunk_indices[1], 0:438] 
									if type(guesses2[q]) is np.ndarray 
									else guesses2[q] 
									for q in range(len(guesses2))]
					limits2 = [limits2[q][chunk_indices[0]:chunk_indices[1], 0:438] 
									if type(limits2[q]) is np.ndarray 
									else limits2[q] 
									for q in range(len(limits2))]
					ties2 = [ties2[q][chunk_indices[0]:chunk_indices[1], 0:438] 
									if type(ties2[q]) is np.ndarray 
									else ties2[q] 
									for q in range(len(ties2))]
				
				# grab specific pixel value from the array
				total_guesses2 = [guesses2[q][j,i] 
									if type(guesses2[q]) is np.ndarray 
									else guesses2[q] 
									for q in range(len(guesses2))]
				
				total_limits2 =  [(limits2[q][0][j,i], limits2[q][1][j,i]) 
										if type(limits2[q][0]) is np.ndarray 
										else limits2[q] 
										for q in range(len(limits2))]
				
				# if the model is nan, then we gotta skip the pixel
				if np.isfinite(total_guesses2[1]) == False:
					if chunk_num == 2 or multiprocess == 1: pbar.update(1)
					count+=1
					continue
				elif 'nan' in ties2[j][i][1]:
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
										tied = ties2[j][i],
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
					amps2_list.append(spec2.measurements.lines[line]['amp']/(1e-20))
					centers2_list.append(spec2.measurements.lines[line]['pos'])
					widths2_list.append(spec2.measurements.lines[line]['fwhm']/2.355)
					
				# calculate reduced chi-square; first add up each Gaussian
				components2 = [one_gaussian(np.array(chunk.spectral_axis), 
								amps2_list[i], centers2_list[i], widths2_list[i]) 
								for i in np.arange(len(amps2_list))]
				model2 = sum(components2)
				redchisq2 = red_chisq(spectrum, model2, num_params=len(amps2_list)*3, 
										err=errs2, free_params=free_params2)
				
				# check for proper component order 
				# increasing wavelength
				# save everything to the parameter cube!!!!
				params2 = [par for sublist in zip(amps2_list, centers2_list, widths2_list)
							for par in sublist]
				ordered_params2 = component_order_check(params2)
				parcube2[:,j,i] = ordered_params2 + redchisq2 # tack redchisq on to end
				
				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plot_one_fit(i, j, spec2, redchisq2, 
									savepath = '%s/fits2/' % savepath, 
									xmin=6530, xmax=6620,
									ymax=max(spectrum), fluxnorm=1e-20,
									input_params = total_guesses2)
				
			# up the counter + progress bar
			count += 1
			if chunk_num == 2 or multiprocess == 1: pbar.update(1)    
			## TODO: GENERALIZE FLUXNORM, XMIN, AND XMAX ABOVE


		############################# 
		## SAVE ONE COMPONENT FIT ###
		############################# 
		
		if fit1 == True:
			# make a silly little header
			hdr1 = fits.Header()
			hdr1['FITTYPE'] = 'gaussian'
			parname = ['Amplitude', 'Center', 'Width'] * int(npars1 // 2 // 3)
			jj = 0
			for ii in range(len(parname)):
				if ii % 3 == 0:
					jj+=1
				kw = "PLANE%i" % ii
				hdr1[kw] = parname[ii] + str(jj)
			hdul1 = fits.PrimaryHDU(data=parcube1, header=hdr1)
				
			try:
				hdul1.writeto('%s/fit1_%s.fits' % (savepath, chunk_num), overwrite=True)
			except:
				hdul1.writeto('fit1_%s.fits' % chunk_num, overwrite=True)


		############################# 
		## SAVE TWO COMPONENT FIT ###
		############################# 

		if fit2 == True:

			# make a silly little header
			hdr2 = fits.Header()
			hdr2['FITTYPE'] = 'gaussian'
			parname = ['Amplitude', 'Center', 'Width'] * int(npars2 // 3)
			jj = 0
			for ii in range(len(parname)):
				if ii % 3 == 0:
					jj+=1
				kw = "PLANE%i" % ii
				hdr2[kw] = parname[ii] + str(jj)
			hdul2 = fits.PrimaryHDU(data=parcube2, header=hdr2)
				
			try:
				hdul2.writeto('%s/fit2_%s.fits' % (savepath, chunk_num), overwrite=True)
			except:
				hdul2.writeto('fit2_%s.fits' % chunk_num, overwrite=True)

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

