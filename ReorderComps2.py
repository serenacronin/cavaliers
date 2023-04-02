"""
Created on Wed Mar 15 2023

@author: Serena A. Cronin

This script will reorder the components of the two Gaussian fit.

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


def reorder_components2(infile, outfile, infile_err, outfile_err):
    
    """
    This function reads in the parameter (infile) and error (infile_err) files of the two Gaussian component fits
    and reorders them based on increasing wavelength. It then ouputs these as reordered parameter (outfile)
    and reordered error (outfile_err) files.
    """

    print('Reordering components (parameters)....')
    ##############################################################################
    # params
    ##############################################################################
    
    # read in fits2
    fits2 = pd.read_csv(infile)

    # open a new file, add in the headers
    f2 = open(outfile, "w")
    f2.write('X,Y,RedChiSq,Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,')
    f2.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,')
    f2.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10\n')

    # reorder the parameters based on wavelength
    sort_index = np.array(np.argsort(fits2.iloc[:,13:23]))
    wvl_arr = np.array(fits2.iloc[:,13:23])
    amps_arr = np.array(fits2.iloc[:,3:13])
    sigs_arr = np.array(fits2.iloc[:,23:33])

    # sort the file based on the reordering above
    for pix in tqdm(range(len(sort_index))):

        # for some reason it is an ndarray
        wvl_params = [wvl_arr[pix][i] for i in sort_index[pix]]
        amps_params = [amps_arr[pix][i] for i in sort_index[pix]]
        sigs_params = [sigs_arr[pix][i] for i in sort_index[pix]]
        ordered_params2 = amps_params + wvl_params + sigs_params  # all sorted params

        # write to a file (saves a TON of time rather than saving to memory)
        f2.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
                '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
                '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' %
                 (fits2['X'][pix], fits2['Y'][pix], fits2['RedChiSq'][pix],
                 ordered_params2[0], ordered_params2[1], ordered_params2[2], ordered_params2[3], ordered_params2[4],
                 ordered_params2[5], ordered_params2[6], ordered_params2[7], ordered_params2[8], ordered_params2[9],
                 ordered_params2[10], ordered_params2[11], ordered_params2[12], ordered_params2[13], ordered_params2[14],
                 ordered_params2[15], ordered_params2[16], ordered_params2[17], ordered_params2[18], ordered_params2[19],
                 ordered_params2[20], ordered_params2[21], ordered_params2[22], ordered_params2[23], ordered_params2[24],
                 ordered_params2[25], ordered_params2[26], ordered_params2[27], ordered_params2[28], ordered_params2[29]))

    f2.close()

    ##############################################################################
    # errors on the params
    ##############################################################################

    print('Reordering component (errors)....')
    # read in errs_fits2
    fits2_err = pd.read_csv(infile_err)
    
    # open a new file, add in the headers
    e2 = open(outfile_err, "w")
    e2.write('X,Y,RedChiSq,Amp1,Amp2,Amp3,Amp4,Amp5,Amp6,Amp7,Amp8,Amp9,Amp10,')
    e2.write('Wvl1,Wvl2,Wvl3,Wvl4,Wvl5,Wvl6,Wvl7,Wvl8,Wvl9,Wvl10,')
    e2.write('Sig1,Sig2,Sig3,Sig4,Sig5,Sig6,Sig7,Sig8,Sig9,Sig10\n')

    # use the new order from above: sort_index
    wvl_arr = np.array(fits2_err.iloc[:,13:23])
    amps_arr = np.array(fits2_err.iloc[:,3:13])
    sigs_arr = np.array(fits2_err.iloc[:,23:33])

    # do the sorting
    for pix in tqdm(range(len(sort_index))):

        # for some reason it is an ndarray
        wvl_params = [wvl_arr[pix][i] for i in sort_index[pix]]

        amps_params = [amps_arr[pix][i] for i in sort_index[pix]]
        sigs_params = [sigs_arr[pix][i] for i in sort_index[pix]]
        ordered_params2 = amps_params + wvl_params + sigs_params  # all sorted errors on params

        # write to a file (saves a TON of time rather than saving to memory)
        e2.write('%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
                '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,'
                '%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n' %
                 (fits2_err['X'][pix], fits2_err['Y'][pix], fits2_err['RedChiSq'][pix],
                ordered_params2[0], ordered_params2[1], ordered_params2[2], ordered_params2[3], ordered_params2[4],
                 ordered_params2[5], ordered_params2[6], ordered_params2[7], ordered_params2[8], ordered_params2[9],
                 ordered_params2[10], ordered_params2[11], ordered_params2[12], ordered_params2[13], ordered_params2[14],
                 ordered_params2[15], ordered_params2[16], ordered_params2[17], ordered_params2[18], ordered_params2[19],
                 ordered_params2[20], ordered_params2[21], ordered_params2[22], ordered_params2[23], ordered_params2[24],
                 ordered_params2[25], ordered_params2[26], ordered_params2[27], ordered_params2[28], ordered_params2[29]))

    e2.close()

    return


def add_velocities2(infile, outfile, restwls, Vsys):
    
    """
    This function converts to velocity the wavelengths of the two Gaussian component fits.
    We can run this function for both the parameter file and the error file.
    """

    print('Adding velocities....')
    
    # read in the file
    outputs_ordered = pd.read_csv(infile, delimiter=',', index_col=False)

    # make velocity columns
    outputs_ordered['Vel1'] = wavelength_to_velocity(outputs_ordered['Wvl1'], Vsys, restwls[0])
    outputs_ordered['Vel2'] = wavelength_to_velocity(outputs_ordered['Wvl2'], Vsys, restwls[0])
    outputs_ordered['Vel3'] = wavelength_to_velocity(outputs_ordered['Wvl3'], Vsys, restwls[1])
    outputs_ordered['Vel4'] = wavelength_to_velocity(outputs_ordered['Wvl4'], Vsys, restwls[1])
    outputs_ordered['Vel5'] = wavelength_to_velocity(outputs_ordered['Wvl5'], Vsys, restwls[2])
    outputs_ordered['Vel6'] = wavelength_to_velocity(outputs_ordered['Wvl6'], Vsys, restwls[2])
    outputs_ordered['Vel7'] = wavelength_to_velocity(outputs_ordered['Wvl7'], Vsys, restwls[3])
    outputs_ordered['Vel8'] = wavelength_to_velocity(outputs_ordered['Wvl8'], Vsys, restwls[3])
    outputs_ordered['Vel9'] = wavelength_to_velocity(outputs_ordered['Wvl9'], Vsys, restwls[4])
    outputs_ordered['Vel10'] = wavelength_to_velocity(outputs_ordered['Wvl10'], Vsys, restwls[4])


    # output to file
    # will overwrite the above but that's fine
    outputs_ordered.to_csv(outfile, index=False)
    
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