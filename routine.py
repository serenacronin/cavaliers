#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
Name: routine.py

Created on Fri Jan 13 10:16:16 2023

Author: Serena A. Cronin

This script is simultaneously fits one, two, and three systems of lines
to each spectrum in an IFU datacube. Each component is modeled as a Gaussian.
========================================================================================
"""

import os
import numpy as np
import pyspeckit
from astropy import units as u
from spectral_cube import SpectralCube
from tqdm import tqdm
from plot_fits import plotting


def gaussian(x_array, amp, cen, sigma):

    """
    Function to create a Gaussian.

    Parameters
	-------------------
	
	x_array : array
        Array of x values.
	
	amp : int or float
        Amplitude of the Gaussian.
	
	cen : int or float
        Center of the Gaussian.
	
	sigma : int or float
        Width of the Gaussian.
	
    """

    return(amp*(np.exp((-1.0/2.0)*(((x_array-cen)/sigma)**2))))



def red_chisq(obs, calc, err, free_params):

    """
    Function to calculate a reduced chi-square.

    Parameters
	-------------------
	
	obs : array
        Array of observed values.
	
	calc : array
        Array of calculated values (i.e., a model).
	
	err : array, int, or float
        Err
	
	sigma : int or float
        Width of the Gaussian.
	
    """

    return((np.sum((obs-calc)**2 / err**2))/(len(obs)-free_params))


def optical_vel_to_ang(vels, Vsys, restwl):
	
	"""
	Quick function to convert optical velocities
	(in km/s) to wavelengths (in Angstroms).
	
	Parameters
	-------------------
	
	vels : float, int, or array
		One or more velocities in km/s to convert
		to wavelength(s) in Angstroms.
		
	Vsys : float or int
        Systemic velocity of the galaxy disk in km/s.
		
	restwl : float or int
		Rest wavelength of the spectral line in Angstroms.
		
	
	"""
	
	c = 3.0*10**5  # speed of light in km/s
	wls = restwl*((vels + Vsys) + c)/c  # conversion equation
	
	return wls


def CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
				ContLower2, ContUpper2):
	
	"""" 
	This will create a baseline-subtracted datacube centered on the 
	wavelengths we want to look at.
	
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
		
	"""

	# read in the full datacube
	cube = SpectralCube.read(filename, hdu=1).spectral_slab(SlabLower * u.AA, 
																SlabUpper * u.AA)

	# blank out the emission lines by masking the cube so that we only
    # use "good" channels, i.e., channels with no emission lines
	spectral_axis = cube.spectral_axis  # grab the spectral axis
	good_channels = ((spectral_axis < ContUpper1*u.AA) |
					 (spectral_axis > ContLower1*u.AA) |
					 (spectral_axis < ContUpper2*u.AA) |
					 (spectral_axis > ContLower2*u.AA))
	masked_cube = cube.with_mask(good_channels[:, np.newaxis, np.newaxis]) 


	# take the median of the remaining continuum (i.e., line-free channels) 
    # and subtract the baseline from the cube
	med = masked_cube.median(axis=0)
	med_cube = cube - med
	cube_final = med_cube.spectral_slab(SlabLower*u.AA, SlabUpper*u.AA)
	return cube_final



def InputParams(fit1, fit2, fit3, R, free_params, continuum_limits, fluxnorm,
				amps1=False, centers1=False, ties1=False, 
				amps2=False, centers2=False, ties2=False,
				amps3=False, centers3=False, ties3=False,
				redchisq_range=False, save_fits=False, savepath=False):
	
	
	"""
	This function will compile your inital guesses for fitting a system of
	one, two, or three lines to IFU data, assuming each component is Gaussian.
    At its simplest, it will gather together your amplitude and
	center wavelength guesses. It also requires the resolving power R of the
	instrument to calculate the widths and an option to tie the fits of multiple 
	emission lines to each other. Note that the centers stem from a velocity model 
    of the galaxy disk.
	
	Parameters
	-------------------
	
	fit1 : bool
		Toggle True if you want to fit with one system of lines.

	fit2 : bool
		Toggle True if you want to fit with two systems of lines.

	fit3 : bool
		Toggle True if you want to fit with three systems of lines.

	R : int or float
		Resolving power of the instrument.

	free_params : int
		Number of free parameters for the fit in order to calculate the
		reduced chi square.

	continuum_limits : list
		Compute the rms for the reduced chi square.
		These are the lower and upper limits of the good channels that are 
		blanked before taking the root mean square of the leftover continuum.
		Format is, e.g., [5000, 6000].

	fluxnorm : int or float
		The flux to normalize the y-axis to.

	amps1 : list of ints or floats; default=False
		Input guesses for the one system of lines.
	
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
		Input guesses for the two systems of lines.
	
	centers2 : string; default=False
		Center wavelength guesses for the two systems of lines.

	ties2 : list of strings; default=False
		See ties_1.

	amps3 : list of ints or floats; default=False
		Input guesses for the three systems of lines.
	
	centers3 : string; default=False
		Center wavelength guesses for the three systems of lines.

	ties3 : list of strings; default=False
		See ties_1.

	redchisq_range : False or string; default=False
		Option to focus on a range of the spectrum for the reduced
		chi square calculation.
		
	save_fits : False or int; default=False
		Trigger to printout of a spectrum and its fit. 
		Will result in a .png file of every n pixel.
		If not False, save_failed_fits = n.
		
	savepath : string; default=False
		Location to save the failed fits.

	"""
	
	# checks to make sure the inputted values are in the ideal format
	if fit1 != False and fit1 != True:
		raise Exception('fit1 must be True or False.')
	if fit2 != False and fit2 != True:
		raise Exception('fit2 must be True or False.')
	if fit3 != False and fit3 != True:
		raise Exception('fit3 must be True or False.')
	if fit1 == True and ((amps1 or centers1 or ties1) == False):
		raise Exception('If fit1 is True, need amps1, centers1, and ties1.')
	if fit2 == True and ((amps2 or centers2 or ties2) == False):
		raise Exception('If fit2 is True, need amps2, centers2, and ties2.')
	if fit3 == True and ((amps3 or centers3 or ties3) == False):
		raise Exception('If fit3 is True, need amps3, centers3, and ties3.')

	if fit1 == True and free_params == False:
		raise Exception('Need the number of free parameters of the fits.')
	if fit2 == True and free_params == False:
		raise Exception('Need the number of free parameters of the fits.')
	if fit3 == True and free_params == False:
		raise Exception('Need the number of free parameters of the fits.')
	if save_fits == True and free_params == False:
		raise Exception('Need the number of free parameters of the fits.')
		
	if save_fits != False and type(save_fits) != int:
		raise Exception('save_fits must be False or int.')
		
	if (fit1 == True) and (continuum_limits == False):
		raise Exception('''To compute the errors for a chi square, we need 
						 lower and upper limits of good channels.''')
	elif (fit2 == True) and (continuum_limits == False):
		raise Exception('''To compute the errors for a chi square, we need 
						 lower and upper limits of good channels.''')
	elif (fit3 == True) and (continuum_limits == False):
		raise Exception('''To compute the errors for a chi square, we need 
						 lower and upper limits of good channels.''')

	if (save_fits == True) and (savepath == False):
		raise Exception('''Please input a savepath for the 
						 save fits flag.''')
                         
	if (fit1 == True) and (fit2 == True):
		raise Exception('Run only one model at a time.')
	elif (fit1 == True) and (fit3 == True):
		raise Exception('Run only one model at a time.')
	elif (fit2 == True) and (fit3 == True):
		raise Exception('Run only one model at a time.')
	elif (fit1 == True) and (fit2 == True) and (fit3 == True):
		raise Exception('Run only one model at a time.')

	# =====================================================================
	# How to Calculate the Widths:
	#
	# Get the lower limit of the widths, which is the spectral resolution.
	# Multiply this by 5 to get the upper limit (5 sigma).
	# THEN take the average of the lower limits, multiply by 3
	# to get our initial guess (3 sigma).
	# Recall: sigma = (wavelength / R) / 2.355 because FWHM = 2.355*sigma
	# ======================================================================

	# do we want a fit where there is only one system of lines?
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

	# if we don't want one system of lines, set things to False	
	else:
		guesses1 = False
		limits1 = False
		limited1 = False
		ties1 = False
		
	# do we want a fit where there are two systems of lines?
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
		
	# do we want a fit where there are three systems of lines?
	# same steps as above
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
	return([fit1, fit2, fit3, free_params, continuum_limits, fluxnorm,
			guesses1, limits1, limited1, ties1,
			guesses2, limits2, limited2, ties2,
			guesses3, limits3, limited3, ties3,
			redchisq_range, save_fits, savepath])


def compute_rms(spec_axis, spectrum, ContLower, ContUpper):
	
	"""
	Take in a spectrum, blank out the emission lines, and calculate the
	root mean square (rms) of the leftover continuum to get the uncertainty
	on the observed values.

	Parameters
	-------------------

	spec_axis : array or list
		Spectral axis in wavelengths.

	spectrum : array or list
		y-axis values of the spectrum.

	ContLower : int or float
		Lower limit of the continuum to blank emission lines.

	ContUpper : int or float
		Upper limit of the continuum to blank emission lines.
	
	"""
	
	# blank out the emission lines to get the continuum
	cont_channels = np.where((spec_axis > ContUpper) |
					 (spec_axis < ContLower))
	
	continuum_vals = spectrum[cont_channels]
	
	# calculate the root mean square of the continuum
	# this will be our uncertainty
	rms = np.sqrt(np.mean(np.square(continuum_vals)))
	
	return rms


def component_order_check(params, fit2 = False, fit3 = False):
	
	
	"""

	A very inelegant check to make sure the redshifted and blueshifted
	components are not getting switched. Not sure why the fitter
	wants to do that in some regions (high S/N, where blueshifted 
									  component is zero)

	Parameters
	-------------------

	params : list
		List of parameters from the main fitting routine.

	fit2 : bool; default=False
		Toggle True for two system of lines.

	fit3 : bool; default=False
		Toggle True for three system of lines.


	"""
	
	# re-order components for two systems of lines
	if fit2 == True:

		# if the parameters are out of order...
		if (params[4] < params[1]) | (params[10] < params[7]) | (params[16] < params[13]) | (params[22] < params[19]) |(params[28] < params[25]):
			
			new_params = np.zeros(len(params))  # create an array to store the reordered parameters

			# first emission line
			a_comp1 = params[0]
			v_comp1 = params[1]
			s_comp1 = params[2]
			
			a_comp2 = params[3]
			v_comp2 = params[4]
			s_comp2 = params[5]
			
			# store the parameters in the correct order
			new_params[3], new_params[4], new_params[5] = a_comp1, v_comp1, s_comp1
			new_params[0], new_params[1], new_params[2] = a_comp2, v_comp2, s_comp2
			
			# second emission line
			a_comp1 = params[6]
			v_comp1 = params[7]
			s_comp1 = params[8]
			
			a_comp2 = params[9]
			v_comp2 = params[10]
			s_comp2 = params[11]
			
			# store the parameters in the correct order
			new_params[9], new_params[10], new_params[11] = a_comp1, v_comp1, s_comp1
			new_params[6], new_params[7], new_params[8] = a_comp2, v_comp2, s_comp2
			
			# third emission line
			a_comp1 = params[12]
			v_comp1 = params[13]
			s_comp1 = params[14]
			
			a_comp2 = params[15]
			v_comp2 = params[16]
			s_comp2 = params[17]

			# store the parameters in the correct order
			new_params[15], new_params[16], new_params[17] = a_comp1, v_comp1, s_comp1
			new_params[12], new_params[13], new_params[14] = a_comp2, v_comp2, s_comp2

			# fourth emission line
			a_comp1 = params[18]
			v_comp1 = params[19]
			s_comp1 = params[20]
			
			a_comp2 = params[21]
			v_comp2 = params[22]
			s_comp2 = params[23]
			
			# store the parameters in the correct order
			new_params[21], new_params[22], new_params[23] = a_comp2, v_comp2, s_comp2
			new_params[18], new_params[19], new_params[20] = a_comp1, v_comp1, s_comp1

			# fifth emission line
			a_comp1 = params[24]
			v_comp1 = params[25]
			s_comp1 = params[26]
			
			a_comp2 = params[27]
			v_comp2 = params[28]
			s_comp2 = params[29]
			
			# store the parameters in the correct order
			new_params[27], new_params[28], new_params[29] = a_comp2, v_comp2, s_comp2
			new_params[24], new_params[25], new_params[26] = a_comp1, v_comp1, s_comp1

		else:
			new_params = params

		# re-order components for three systems of lines
		if fit3 == True:

			# if the parameters are out of order...
			if (params[4] < params[1]) | (params[10] < params[7]) | (params[16] < params[13]):
				new_params = np.zeros(len(params))  # create an array to store the reordered parameters

				# first emission line
				a_comp1 = params[0]
				v_comp1 = params[1]
				s_comp1 = params[2]
				
				a_comp2 = params[3]
				v_comp2 = params[4]
				s_comp2 = params[5]
				
				# store the parameters in the correct order
				new_params[3], new_params[4], new_params[5] = a_comp1, v_comp1, s_comp1
				new_params[0], new_params[1], new_params[2] = a_comp2, v_comp2, s_comp2
				
				# second emission line
				a_comp1 = params[6]
				v_comp1 = params[7]
				s_comp1 = params[8]
				
				a_comp2 = params[9]
				v_comp2 = params[10]
				s_comp2 = params[11]
				
				# store the parameters in the correct order
				new_params[9], new_params[10], new_params[11] = a_comp1, v_comp1, s_comp1
				new_params[6], new_params[7], new_params[8] = a_comp2, v_comp2, s_comp2
				
				# third emission line
				a_comp1 = params[12]
				v_comp1 = params[13]
				s_comp1 = params[14]
				
				a_comp2 = params[15]
				v_comp2 = params[16]
				s_comp2 = params[17]
				
				# store the parameters in the correct order
				new_params[15], new_params[16], new_params[17] = a_comp1, v_comp1, s_comp1
				new_params[12], new_params[13], new_params[14] = a_comp2, v_comp2, s_comp2

				# fourth emission line
				a_comp1 = params[18]
				v_comp1 = params[19]
				s_comp1 = params[20]
				
				a_comp2 = params[21]
				v_comp2 = params[22]
				s_comp2 = params[23]
				
				# store the parameters in the correct order
				new_params[21], new_params[22], new_params[23] = a_comp2, v_comp2, s_comp2
				new_params[18], new_params[19], new_params[20] = a_comp1, v_comp1, s_comp1

				# fifth emission line
				a_comp1 = params[24]
				v_comp1 = params[25]
				s_comp1 = params[26]
				
				a_comp2 = params[27]
				v_comp2 = params[28]
				s_comp2 = params[29]
				
				# store the parameters in the correct order
				new_params[27], new_params[28], new_params[29] = a_comp2, v_comp2, s_comp2
				new_params[24], new_params[25], new_params[26] = a_comp1, v_comp1, s_comp1	
			
	return new_params


def FitRoutine(FittingInfo, cube):
	
	"""
	This function wraps around the pyspeckit and spectral-cube packages
	to fit one, two, and three systems of lines to spectra in an IFU datacube,
	assuming each component is Gaussian. The best way to use this is in 
	conjunction with the InputParams() function, which will generate the initial 
	guesses for the Gaussian models.
	
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
			FittingInfo[5] : fluxnorm
			FittingInfo[6] : guesses1
			FittingInfo[7] : limits1
			FittingInfo[8] : limited1
			FittingInfo[9] : ties1
			FittingInfo[10] : guesses2
			FittingInfo[11] : limits2
			FittingInfo[12] : limited2
			FittingInfo[13] : ties2
			FittingInfo[14] : guesses3
			FittingInfo[15] : limits3
			FittingInfo[16] : limited3
			FittingInfo[17] : ties3
			FittingInfo[18] : save_fits
			FittingInfo[19] : savepath

	cube : spectral-cube object
		Datacube from CreateCube() function.
			
		See InputParams() for more details.
		
	"""

	# ============================================================================================================
	# PREPARE FILES FOR THE FITTING ROUTINE
	# ============================================================================================================

	# grab all the input parameters!
	fit1 = FittingInfo[0]
	fit2 = FittingInfo[1]
	fit3 = FittingInfo[2]
	free_params = FittingInfo[3]
	continuum_limits = FittingInfo[4]
	fluxnorm = FittingInfo[5]
	guesses1 = FittingInfo[6]
	limits1 = FittingInfo[7]
	limited1 = FittingInfo[8]
	ties1 = FittingInfo[9]
	guesses2 = FittingInfo[10]
	limits2 = FittingInfo[11]
	limited2 = FittingInfo[12]
	ties2 = FittingInfo[13]
	guesses3 = FittingInfo[14]
	limits3 = FittingInfo[15]
	limited3 = FittingInfo[16]
	ties3 = FittingInfo[17]
	save_fits = FittingInfo[18]
	savepath = FittingInfo[19]
	
	# make directories to store the .png files of the fits
	if (save_fits != False) & (fit1 == True):
		if not os.path.exists('%sfits1/' % savepath):
			os.makedirs('%sfits1/' % savepath)

	if (save_fits != False) & (fit2 == True):
		if not os.path.exists('%sfits2/' % savepath):
			os.makedirs('%sfits2/' % savepath)

	if (save_fits != False) & (fit3 == True):
		if not os.path.exists('%sfits3/' % savepath):
			os.makedirs('%sfits3/' % savepath)

	# make text files to output all of our fitted parameters
	if fit1 == True:

		# first, remove the file if it exists so we can overwrite w new lines
		if os.path.exists("%sfits1.txt" % savepath):
			os.remove("%sfits1.txt" % savepath)

		# now create the file with headers
		f1 = open("%sfits1.txt" % savepath, "w")
		f1.write('X,Y,RedChiSq,')
		f1.write('Amp1,Amp2,Amp3,Amp4,Amp5,')
		f1.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,')
		f1.write('Sig1,Sig2,Sig3,Sig4,Sig5\n')

		# save errors to another file
		if os.path.exists("%sfits1_err.txt" % savepath):
			os.remove("%sfits1_err.txt" % savepath)

		e1 = open("%sfits1_err.txt" % savepath, "w")
		e1.write('X,Y,RedChiSq,')
		e1.write('Amp1,Amp2,Amp3,Amp4,Amp5,')
		e1.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,')
		e1.write('Sig1,Sig2,Sig3,Sig4,Sig5\n')

	# same as above, but for the two systems of lines model
	if fit2 == True:

		if os.path.exists("%sfits2.txt" % savepath):
			os.remove("%sfits2.txt" % savepath)

		f2 = open("%sfits2.txt" % savepath, "w")
		f2.write('X,Y,RedChiSq,')
		f2.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,')
		f2.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,')
		f2.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10\n')

		if os.path.exists("%sfits2_err.txt" % savepath):
			os.remove("%sfits2_err.txt" % savepath)

		e2 = open("%sfits2_err.txt" % savepath, "w")
		e2.write('X,Y,RedChiSq,')
		e2.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,')
		e2.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,')
		e2.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10\n')


	# same as above, but for the three systems of lines model
	if fit3 == True:
		if os.path.exists("%sfits3.txt" % savepath):
			os.remove("%sfits3.txt" % savepath)

		f3 = open("%sfits3.txt" % savepath, "w")
		f3.write('X,Y,RedChiSq,')
		f3.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,Amp11,Amp12,Amp13,Amp14,Amp15')
		f3.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,Wvl11,Wvl12,Wvl13,Wvl14,Wvl15')
		f3.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10,Sig11,Sig12,Sig13,Sig14,Sig15\n')

		if os.path.exists("%sfits3_err.txt" % savepath):
			os.remove("%sfits3_err.txt" % savepath)

		e3 = open("%sfits3_err.txt" % savepath, "w")
		e3.write('X,Y,RedChiSq,')
		e3.write('Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,Amp11,Amp12,Amp13,Amp14,Amp15')
		e3.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,Wvl11,Wvl12,Wvl13,Wvl14,Wvl15')
		e3.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10,Sig11,Sig12,Sig13,Sig14,Sig15\n')
	
	# ============================================================================================================
	# PERFORM THE FITTING ROUTINE
	# ============================================================================================================

	# create two parameter cubes for the fits
	# z-dimension is npars + 1 because we want an
	# extra parameter to save the reduced chi square values
	_, y, x = cube.shape

	# set up a progress bar and a count of the pixels
	pbar = tqdm(total=x*y, desc='Running fitting routine...')
	count = 0

	# loop over each pixel and run the fitting routine!
	for i in np.arange(x): # x-axis 
		for j in np.arange(y): # y-axis

			spectrum = np.array(cube[:,j,i], dtype='float64')  # grab the spectrum
						
			# if we land on a nan pixel, skip and update the progressbar and the count
			if np.isfinite(np.mean(spectrum)) == False:
				pbar.update(1)  
				count+=1
				continue

			# grab spectral axis within our desired wavelength range
			minval = min(np.array(cube.spectral_axis))
			maxval = max(np.array(cube.spectral_axis))
			spec_axis = np.linspace(minval, maxval, len(spectrum))
				
			
			# ============================================================================================================ 
			# ONE SYSTEM OF LINES
			# ============================================================================================================
			if fit1 == True:

				spec1 = pyspeckit.Spectrum(data=spectrum, xarr=spec_axis) # grab the spectrum

				# grab specific pixel value from the array
				total_guesses1 = [guesses1[q][j,i] 
									if type(guesses1[q]) == np.ndarray 
									else guesses1[q]
									for q in range(len(guesses1))]
				
				total_limits1 = [(limits1[q][0][j,i], limits1[q][1][j,i]) 
								if type(limits1[q][0]) == np.ndarray 
								else limits1[q] 
								for q in range(len(limits1))]

				# if the model is nan, then skip the pixel and update the progressbar and count
				if np.isfinite(total_guesses1[1]) == False:
					pbar.update(1)
					count+=1
					continue
					
				# perform the fit!
				spec1.specfit.multifit(fittype='gaussian',
										guesses = total_guesses1, 
										limits = total_limits1,
										limited = limited1,
										tied = ties1,
										annotate = False)
				spec1.measure(fluxnorm = fluxnorm)
				
				# get errors on the spectrum for the reduced chi square
				errs1 = compute_rms(spec_axis, spectrum, continuum_limits[0], 
									continuum_limits[1])

				# get fit params
				amps1_list = []
				centers1_list = []
				widths1_list = []
				for line in spec1.measurements.lines.keys():
					amps1_list.append(round(spec1.measurements.lines[line]['amp']/(fluxnorm), 4)) #TODO: GENERALIZE
					centers1_list.append(round(spec1.measurements.lines[line]['pos'], 4))
					widths1_list.append(round(spec1.measurements.lines[line]['fwhm']/2.355,4))

				# grab the error on each parameter
				err_params1 = [round(e,4) for e in spec1.specfit.parinfo.errors]

				# save the parameters to a list
				params1 = [par for sublist in zip(amps1_list, centers1_list, widths1_list)
							for par in sublist]
			
				# calculate the reduced chi square
				components1 = [gaussian(np.array(cube.spectral_axis), 
								amps1_list[i], centers1_list[i], widths1_list[i]) for i in
								np.arange(len(amps1_list))]
				model1 = sum(components1)

				# what we want is to do the reduced chi square only over the range of emission lines
				# bc we have such a long baseline between emission lines
				# FIXME: generalize this so the user can input the wavelength ranges				
				chans_ind = np.argwhere(((np.array(cube.spectral_axis) > 6525) & (np.array(cube.spectral_axis) < 6620)) |
						((np.array(cube.spectral_axis) > 6700) & (np.array(cube.spectral_axis) < 6750)))
				
				# grab the channels we want from the spectrum itself and our model
				chans_spec = np.array(spectrum)[chans_ind]
				chans_model1 = model1[chans_ind]

				redchisq1 = round(red_chisq(chans_spec, chans_model1, 
					num_params=len(amps1_list)*3, err=errs1, 
					free_params=free_params),4)
				
				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plotting(i, j, spec1, redchisq1, 
									savepath = '%sfits1' % savepath, 
									xmin=6500, xmax=6800, 
									ymax=max(spectrum), fluxnorm=fluxnorm,
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


			# ============================================================================================================ 
			# TWO SYSTEMS OF LINES
			# ============================================================================================================
			if fit2 == True:
				
				# grab specific pixel value from the array
				total_guesses2 = [guesses2[q][j,i] 
									if type(guesses2[q]) == np.ndarray 
									else guesses2[q] 
									for q in range(len(guesses2))]
			
				
				total_limits2 =  [(limits2[q][0][j,i], limits2[q][1][j,i]) 
										if type(limits2[q][0]) == np.ndarray 
										else limits2[q] 
										for q in range(len(limits2))]
				
				# if the model is nan, then we skip the pixel and update the progressbar and count
				if np.isfinite(total_guesses2[1]) == False:
					pbar.update(1)
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
										tied = ties2,
										annotate = False)
				spec2.measure(fluxnorm = fluxnorm)
				
				# get errors for the reduced chi square
				errs2 = compute_rms(spec_axis, spectrum, continuum_limits[0], 
									continuum_limits[1])
				
				# get the fit params
				amps2_list = []
				centers2_list = []
				widths2_list = []
				for line in spec2.measurements.lines.keys():
					amps2_list.append(round(spec2.measurements.lines[line]['amp']/(fluxnorm),4))
					centers2_list.append(round(spec2.measurements.lines[line]['pos'],4))
					widths2_list.append(round(spec2.measurements.lines[line]['fwhm']/2.355,4))

				# grab the error on each parameter
				err_params2 = [round(e,4) for e in spec2.specfit.parinfo.errors]

				# save all of the parameters to a list
				params2 = [par for sublist in zip(amps2_list, centers2_list, widths2_list)
							for par in sublist]

				# calculate reduced chi-square; first add up each Gaussian
				components2 = [gaussian(np.array(cube.spectral_axis), 
								amps2_list[i], centers2_list[i], widths2_list[i]) 
								for i in np.arange(len(amps2_list))]
				model2 = sum(components2)
				
				# what we want is to do the reduced chi square only over the range of emission lines
				# bc we have such a long baseline between emission lines
				# FIXME: generalize this so the user can input the wavelength ranges			
				chans_ind = np.argwhere(((np.array(cube.spectral_axis) > 6525) & (np.array(cube.spectral_axis) < 6620)) |
						((np.array(cube.spectral_axis) > 6700) & (np.array(cube.spectral_axis) < 6750)))
				
				# grab the channels we want from the spectrum itself and our model
				chans_spec = np.array(spectrum)[chans_ind]
				chans_model2 = model2[chans_ind]

				redchisq2 = round(red_chisq(chans_spec, chans_model2, 
					num_params=len(amps2_list)*3, err=errs2, 
					free_params=free_params),4)

				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plotting(i, j, spec2, redchisq2, 
									savepath = '%sfits2' % savepath, 
									xmin=6500, xmax=6800,
									ymax=max(spectrum), fluxnorm=fluxnorm,
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


			# ============================================================================================================ 
			# THREE SYSTEMS OF LINES
			# ============================================================================================================
			if fit3 == True:
				
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
					pbar.update(1)
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
										tied = ties3,
										annotate = False)
				spec3.measure(fluxnorm = fluxnorm)
				
				# get errors for the reduced chi square
				errs3 = compute_rms(spec_axis, spectrum, continuum_limits[0], 
									continuum_limits[1])
				
				# get the fit params
				amps3_list = []
				centers3_list = []
				widths3_list = []
				for line in spec3.measurements.lines.keys():
					amps3_list.append(round(spec3.measurements.lines[line]['amp']/(fluxnorm),4))
					centers3_list.append(round(spec3.measurements.lines[line]['pos'],4))
					widths3_list.append(round(spec3.measurements.lines[line]['fwhm']/2.355,4))

				# grab the error on each parameter
				err_params3 = [round(e,4) for e in spec3.specfit.parinfo.errors]

				# save the parameters to a list
				params3 = [par for sublist in zip(amps3_list, centers3_list, widths3_list)
							for par in sublist]

				# calculate reduced chi-square; first add up each Gaussian
				components3 = [gaussian(np.array(cube.spectral_axis), 
								amps3_list[i], centers3_list[i], widths3_list[i]) 
								for i in np.arange(len(amps3_list))]
				model3 = sum(components3)
				
				# what we want is to do the reduced chi square only over the range of emission lines
				# bc we have such a long baseline between emission lines
				# FIXME: generalize this so the user can input the wavelength ranges		
				chans_ind = np.argwhere(((np.array(cube.spectral_axis) > 6525) & (np.array(cube.spectral_axis) < 6620)) |
						((np.array(cube.spectral_axis) > 6700) & (np.array(cube.spectral_axis) < 6750)))
				
				# grab the channels we want from the spectrum itself and our model
				chans_spec = np.array(spectrum)[chans_ind]
				chans_model3 = model3[chans_ind]

				redchisq3 = round(red_chisq(chans_spec, chans_model3, 
					num_params=len(amps3_list)*3, err=errs3, 
					free_params=free_params),4)
				
				# option to print out fits
				if (save_fits != False) & (count % save_fits == 0):						
						# print the fit
						plotting(i, j, spec3, redchisq3, 
									savepath = '%s/fits3' % savepath, 
									xmin=6500, xmax=6800,
									ymax=max(spectrum), fluxnorm=fluxnorm,
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
				
			# up the progressbar and count
			count += 1
			pbar.update(1)

	return