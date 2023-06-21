"""
Created on Mon June 6 2023

@author: Serena A. Cronin

This script will reorder the components of the one system of lines fit.

"""
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

##############################################################################
# FUNCTIONS
##############################################################################

def wavelength_to_velocity(wls, Vsys, restwl):

	"""
	This function converts wavelengths to velocity in km/s.
	"""

	c = 3.0*10**5

	vels = (wls*c / restwl) - c

	return vels - Vsys


def reorder_components1(infile, outfile, infile_err, outfile_err):
	
	"""
	This function reads in the parameter (infile) and error (infile_err) files of the two Gaussian component fits
	and reorders them based on increasing wavelength. It then ouputs these as reordered parameter (outfile)
	and reordered error (outfile_err) files.
	"""

	print('Reordering components (parameters)....')
	##############################################################################
	# params
	##############################################################################
	
	# read in fits1
	fits1 = pd.read_csv(infile)

	# open a new file, add in the headers
	f1 = open(outfile, "w")
	f1.write('X,Y,RedChiSq,Amp1,Amp2,Amp3,Amp4,Amp5,')
	f1.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,')
	f1.write('Sig1,Sig2,Sig3,Sig4,Sig5\n')

	# reorder the parameters based on wavelength
	sort_index = np.array(np.argsort(fits1.iloc[:,8:12]))
	wvl_arr = np.array(fits1.iloc[:,8:12])
	amps_arr = np.array(fits1.iloc[:,3:8])
	sigs_arr = np.array(fits1.iloc[:,12:17])

	# sort the file based on the reordering above
	for pix in tqdm(range(len(sort_index))):

		# for some reason it is an ndarray
		wvl_params = [wvl_arr[pix][i] for i in sort_index[pix]]
		amps_params = [amps_arr[pix][i] for i in sort_index[pix]]
		sigs_params = [sigs_arr[pix][i] for i in sort_index[pix]]
		ordered_params1 = amps_params + wvl_params + sigs_params  # all sorted params

		# write to a file (saves a TON of time rather than saving to memory)
		f1.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
				'%s, %s, %s, %s, %s\n' %
				 (fits1['X'][pix], fits1['Y'][pix], fits1['RedChiSq'][pix],
				 ordered_params1[0], ordered_params1[1], ordered_params1[2], ordered_params1[3], ordered_params1[4],
				 ordered_params1[5], ordered_params1[6], ordered_params1[7], ordered_params1[8], ordered_params1[9],
				 ordered_params1[10], ordered_params1[11], ordered_params1[12], ordered_params1[13], ordered_params1[14],
				 ordered_params1[15], ordered_params1[16], ordered_params1[17]))

	f1.close()

	##############################################################################
	# errors on the params
	##############################################################################

	print('Reordering component (errors)....')
	# read in errs_fits2
	fits1_err = pd.read_csv(infile_err)
	
	# open a new file, add in the headers
	e1 = open(outfile_err, "w")
	e1.write('X,Y,RedChiSq,Amp1,Amp2,Amp3,Amp4,Amp5,')
	e1.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,')
	e1.write('Sig1,Sig2,Sig3,Sig4,Sig5\n')

	# use the new order from above: sort_index
	wvl_arr = np.array(fits1_err.iloc[:,8:12])
	amps_arr = np.array(fits1_err.iloc[:,3:8])
	sigs_arr = np.array(fits1_err.iloc[:,12:17])

	# do the sorting
	for pix in tqdm(range(len(sort_index))):

		# for some reason it is an ndarray
		wvl_params = [wvl_arr[pix][i] for i in sort_index[pix]]

		amps_params = [amps_arr[pix][i] for i in sort_index[pix]]
		sigs_params = [sigs_arr[pix][i] for i in sort_index[pix]]
		ordered_params2 = amps_params + wvl_params + sigs_params  # all sorted errors on params

		# write to a file (saves a TON of time rather than saving to memory)
		e1.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
				'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
				'%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' %
				 (fits1_err['X'][pix], fits1_err['Y'][pix], fits1_err['RedChiSq'][pix],
				 ordered_params1[0], ordered_params1[1], ordered_params1[2], ordered_params1[3], ordered_params1[4],
				 ordered_params1[5], ordered_params1[6], ordered_params1[7], ordered_params1[8], ordered_params1[9],
				 ordered_params1[10], ordered_params1[11], ordered_params1[12], ordered_params1[13], ordered_params1[14],
				 ordered_params1[15], ordered_params1[16], ordered_params1[17]))

	e1.close()

	return


def add_velocities1(infile, err_infile, outfile, err_outfile, restwls, Vsys, i):
	
	"""
	This function converts to velocity the wavelengths of the two Gaussian component fits.
	We can run this function for both the parameter file and the error file.
	"""

	print('Adding velocities....')
	
	# read in the file
	outputs_ordered = pd.read_csv(infile, delimiter=',', index_col=False)
	err = pd.read_csv(err_infile, delimiter=',', index_col=False)

	# make velocity columns
	outputs_ordered['Vel1'] = wavelength_to_velocity(outputs_ordered['Wvl1'], Vsys, restwls[0]) / np.sin(i*np.pi/180)
	outputs_ordered['Vel2'] = wavelength_to_velocity(outputs_ordered['Wvl2'], Vsys, restwls[1]) / np.sin(i*np.pi/180)
	outputs_ordered['Vel3'] = wavelength_to_velocity(outputs_ordered['Wvl3'], Vsys, restwls[2]) / np.sin(i*np.pi/180)
	outputs_ordered['Vel4'] = wavelength_to_velocity(outputs_ordered['Wvl4'], Vsys, restwls[3]) / np.sin(i*np.pi/180)
	outputs_ordered['Vel5'] = wavelength_to_velocity(outputs_ordered['Wvl5'], Vsys, restwls[4]) / np.sin(i*np.pi/180)

	# make columns for sigma in velocity space
	outputs_ordered['SigVel1'] = (3*10**5 * outputs_ordered['Sig1']) / restwls[0]
	outputs_ordered['SigVel2'] = (3*10**5 * outputs_ordered['Sig2']) / restwls[1]
	outputs_ordered['SigVel3'] = (3*10**5 * outputs_ordered['Sig3']) / restwls[2]
	outputs_ordered['SigVel4'] = (3*10**5 * outputs_ordered['Sig4']) / restwls[3]
	outputs_ordered['SigVel5'] = (3*10**5 * outputs_ordered['Sig5']) / restwls[4]

	# output to file
	# will overwrite the above but that's fine
	outputs_ordered.to_csv(outfile, index=False)
	
	c = 3.0*10**5

	# do the same for errors
	# using error propagation
	err['Vel1'] = (c / restwls[0])*err['Wvl1']
	err['Vel2'] = (c / restwls[0])*err['Wvl2']
	err['Vel3'] = (c / restwls[1])*err['Wvl3']
	err['Vel4'] = (c / restwls[1])*err['Wvl4']
	err['Vel5'] = (c / restwls[2])*err['Wvl5']
	err['Vel6'] = (c / restwls[2])*err['Wvl6']
	err['Vel7'] = (c / restwls[3])*err['Wvl7']
	err['Vel8'] = (c / restwls[3])*err['Wvl8']
	err['Vel9'] = (c / restwls[4])*err['Wvl9']
	err['Vel10'] = (c / restwls[4])*err['Wvl10']

	err['SigVel1'] = (c / restwls[0])*err['Sig1']
	err['SigVel2'] = (c / restwls[0])*err['Sig2']
	err['SigVel3'] = (c / restwls[1])*err['Sig3']
	err['SigVel4'] = (c / restwls[1])*err['Sig4']
	err['SigVel5'] = (c / restwls[2])*err['Sig5']
	err['SigVel6'] = (c / restwls[2])*err['Sig6']
	err['SigVel7'] = (c / restwls[3])*err['Sig7']
	err['SigVel8'] = (c / restwls[3])*err['Sig8']
	err['SigVel9'] = (c / restwls[4])*err['Sig9']
	err['SigVel10'] = (c / restwls[4])*err['Sig10']
	
	err.to_csv(err_outfile, index=False)
	
	return


def true_errors(infile, err_infile, outfile, err_outfile):
	"""
	This function calculates the true errors by multiplying what we get from
	the fitting program by the rms of the cube.
	"""

	print('Calculating true errors....')
	 
	from CAVALIERS import CreateCube, compute_rms

	filename = '../ngc253/data/ADP.2018-11-22T21_29_46.157.fits'
	infile = pd.read_csv(infile, delimiter=',', index_col=False)
	err_infile = pd.read_csv(err_infile, delimiter=',', index_col=False)

	# info for continuum
	SlabLower = 6500
	SlabUpper = 6800
	ContUpper1 = 6620
	ContLower1 = 6525
	ContUpper2 = 6750
	ContLower2 = 6700

	cube = CreateCube(filename, SlabLower, SlabUpper, ContLower1, ContUpper1,
					ContLower2, ContUpper2, Region=False)

	z, y, x = cube.shape

	minval = min(np.array(cube.spectral_axis))
	maxval = max(np.array(cube.spectral_axis))

	rms_list = []

	for index, row in tqdm(infile.iterrows()):

		i = int(row['X'])
		j = int(row['Y'])

		spectrum = np.array(cube[:,j,i], dtype='float64')
		x_axis = np.linspace(minval, maxval, len(spectrum))
		rms = compute_rms(x_axis, spectrum, ContLower1, ContUpper2)
		rms_list.append(rms)

	# add the rms to the parameter file and error file
	infile['rms'] = rms_list
	err_infile['rms'] = rms_list

	# multiply the errors by the rms
	err_infile.iloc[:,3:-1].multiply(err_infile['rms'], axis="index")

	# err_infile.iloc[:,3:-1] = err_infile.iloc[:,3:-1]*rms_list

	# save to file
	err_infile.to_csv(err_outfile, index=False)
	infile.to_csv(outfile, index=False)

	return



def flux_map2(og, infile, outfile1, outfile2, line):
	
	"""
	This function creates intensity maps for each line in each fit. It produces two maps:
	one for the blueshifted component and the other for the redshifted component.
	"""

	# read in original data
	hdu = fits.open(og)[1]
	og_data = hdu.data
	y, x = og_data[1].shape
	
	# use the original data to create the dimensions
	# of the flux maps
	mapp_blue = np.zeros((y, x))
	mapp_red = np.zeros((y, x))

	# read in fit data
	fits2 = pd.read_csv(infile)

	# generate the flux map(s)
	if line == 'Halpha':
		print('Generating flux map for H-alpha....')
		for index, row in tqdm(fits2.iterrows()):
				mapp_blue[int(row['Y']),int(row['X'])] = row['Amp3']
				mapp_red[int(row['Y']),int(row['X'])] = row['Amp4']

	if line == 'NIIb':
		print('Generating flux map for NIIb....')
		for index, row in tqdm(fits2.iterrows()):
				mapp_blue[int(row['Y']),int(row['X'])] = row['Amp5']
				mapp_red[int(row['Y']),int(row['X'])] = row['Amp6']

	# blank out the edges using the original data
	mapp_blue[np.isnan(og_data[1])] = np.nan # [0] has some nans within
	mapp_red[np.isnan(og_data[1])] = np.nan # [0] has some nans within

	# create fits files to store maps
	hdu2_b = fits.PrimaryHDU(mapp_blue)
	hdu2_r = fits.PrimaryHDU(mapp_red)   
	hdu2_b.writeto(outfile1, overwrite=True)
	hdu2_r.writeto(outfile2, overwrite=True)
		
	return