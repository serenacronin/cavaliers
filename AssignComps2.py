"""
Created on Wed Mar 15 2023

@author: Serena A. Cronin

This script will assign the components of the two Gaussian fit.

"""

# imports
import sys
sys.path.append('../astro_tools')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

# set up future plots
plt.rcParams['text.usetex'] = False
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["axes.labelweight"] = 'bold'
plt.rcParams["axes.titleweight"] = 'bold'
plt.rcParams["font.family"] = "courier new"
plt.rcParams["font.style"] = "normal"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["font.weight"] = 'bold'

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


def Mult_Div_StdErr(x, y, dx, dy):
    """
    Propagation of standard deviation errors for multiplication
    or division. Only two numbers involved.
    """
    return np.sqrt((dx/x)**2 + (dy/y)**2)


# does the two Gaussian component fit work?
def which_fit2(infile, savepath):

    """
    This function determines if the two Gaussian fit is reasonable to use.
    There is an option to make a map of which pixels have "failed" two Gaussian fits.
    """
    print('Checking if the fits passed....')

    # read in file
    fits = pd.read_csv(infile)
    
    # working with H-alpha
    # especially since FWHM should be the same between lines
    
    # define checks
    check1 = []
    check2 = []
    check3 = []
    check4 = []
    
    for index, row in fits.iterrows():
    
        # if H-alpha FWHM is less than the instrumental resolution
        # which is about 2.196 Angstroms
        # actual R = 2989; use next time
        if (row['Sig3']*2.355 < (6562.801 / 3000)) | (row['Sig4']*2.355 < (6562.801 / 3000)):
            check1.append('FAIL')
        else:
            check1.append('PASS')

        # is FWHM greater than ~2 * 450 km/s? (absolute value)
        if ((row['SigVel3'] > 1000) | (row['SigVel4'] > 1000)):
            check2.append('FAIL')
        else:
            check2.append('PASS')

        # are the velocities similar between components?
        # arbitrarily choosing less than 1/2 resolution
        if (np.abs(row['Wvl3'] - row['Wvl4']) < (6562.80 / 3000)/2):
            check3.append('FAIL')
        else:
            check3.append('PASS')

        # are the velocities unphysical?
        # i.e., greater than ~600 km/s (absolute value)
        # vel3 = wavelength_to_velocity(row['Wvl3'], Vsys=243., restwl=6562.801)
        # vel4 = wavelength_to_velocity(row['Wvl4'], Vsys=243., restwl=6562.801)

        if (np.abs(row['Vel3']) > 600) | (np.abs(row['Vel4']) > 600):
            check4.append('FAIL')
        else:
            check4.append('PASS')
            
    # add onto dataframe
    fits['CHECK1'] = check1
    fits['CHECK2'] = check2
    fits['CHECK3'] = check3
    fits['CHECK4'] = check4

    # save to file
    fits.to_csv('%sfits2_reordered.txt' % savepath, index=False)

    #TODO: add plotting option
    
    return fits


def SigToNoise(infile, infile_err, savepath, plot, og):

    """
    This function calculates the signal to noise of the parameters
    to their errors and determines if the S/N is > 5 sigma.
    
    """

    print('Determining signal to noise....')

    par = pd.read_csv(infile)
    err = pd.read_csv(infile_err)

    for num in range(1, 11):

        amp = par['Amp%s' % num] / err['Amp%s' % num]
        wvl = par['Wvl%s' % num] / err['Wvl%s' % num]
        fwhm = par['Sig%s' % num] / err['Sig%s' % num]

        par['CHECK_Amp%s_StoN' % num] = ['FAIL' if i < 5. else 'PASS' for i in amp]
        par['CHECK_Wvl%s_StoN' % num] = ['FAIL' if i < 5. else 'PASS' for i in wvl]
        par['CHECK_Sig%s_StoN' % num] = ['FAIL' if i < 5. else 'PASS' for i in fwhm]

    # collect all into one check
    # i.e., if any parameter fails, we fail the entire pixel
    tot = []
    for index, row in par.iterrows():

        for num in range(1, 11):
            if row['CHECK_Amp%s_StoN' % num] == 'FAIL':
                tot.append('FAIL')
                break
            elif row['CHECK_Wvl%s_StoN' % num] == 'FAIL':
                tot.append('FAIL')
                break
            elif row['CHECK_Sig%s_StoN' % num] == 'FAIL':
                tot.append('FAIL')
                break
            else:
                tot.append('PASS')
                break
        
    par['CHECK_All_StoN'] = tot
    par.to_csv('%sfits2_reordered_S2N.txt' % savepath, index=False)

    # option to plot
    if plot == True:

        # get info of original data
        hdu = fits.open(og)[1]
        og_data = hdu.data
        y, x = og_data[1].shape
        
        # use the original data to create the dimensions of the maps
        mapp = np.zeros((y,x))

        # make a boolean map
        for index, row in par.iterrows():
            if row['CHECK_All_StoN'] == 'FAIL':
                mapp[int(row['Y']), int(row['X'])] = 0
            elif row['CHECK_All_StoN'] == 'PASS':
                mapp[int(row['Y']), int(row['X'])] = 1

        plt.figure(figsize=(7,7))
        ax = plt.subplot(1, 1, 1)
        im = ax.imshow(mapp, origin='lower', cmap='binary')
        ax.set_title('(1) S/N > 5; (0) S/N < 5', fontsize=20)
        ax.set_xlabel('R.A.', fontsize=20)
        ax.set_ylabel('Dec.', fontsize=20)
        bar = plt.colorbar(im, fraction=0.046)
        bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
        plt.savefig('%splots/fits2_SigToNoise.png' % savepath, dpi=200)

    return


# assign the components and create maps!
def assign_comps_mapps(og, infile, outflow, disk, line, criteria, savepath, plot_NII_Halpha):

    # get info of original data
    hdu = fits.open(og)[1]
    og_data = hdu.data
    y, x = og_data[1].shape
    
    # use the original data to create the dimensions of the maps
    mapp_out_amp = np.zeros((y,x))
    mapp_disk_amp = np.zeros((y,x))
    mapp_out_vel = np.zeros((y,x))
    mapp_disk_vel = np.zeros((y,x))
    mapp_out_fwhm = np.zeros((y,x))
    mapp_disk_fwhm = np.zeros((y,x))
    
    if line == 'Halpha':
        
        amp_blue = 'Amp3'
        vel_blue = 'Vel3'
        sig_blue = 'SigVel3'
        amp_red = 'Amp4'
        vel_red = 'Vel4'
        sig_red = 'SigVel4'
    
    if line == 'NIIb':
        
        amp_blue = 'Amp5'
        vel_blue = 'Vel5'
        sig_blue = 'SigVel5'
        amp_red = 'Amp6'
        vel_red = 'Vel6'
        sig_red = 'SigVel6'

    if criteria == 'NII_Halpha_Ratio':
        blue_crit = 'CHECK_BLUE_NII_Halpha_Ratio'
        red_crit = 'CHECK_RED_NII_Halpha_Ratio'

    if criteria == 'Velocities':
        blue_crit = 'CHECK_BLUE_Velocities'
        red_crit = 'CHECK_RED_Velocities'

    if criteria == 'Velocities_Ratio':
        blue_crit = 'CHECK_BLUE_Velocities_Ratio'
        red_crit = 'CHECK_RED_Velocities_Ratio'

    for index, row in infile.iterrows():

        # account for the checks
        # including signal to noise
        if ((row['CHECK1'] == 'FAIL') | (row['CHECK2'] == 'FAIL') | 
            (row['CHECK3'] == 'FAIL') | (row['CHECK4'] == 'FAIL') | 
            (row['CHECK_All_StoN'] == 'FAIL')):
            mapp_out_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.nan
            continue


        if (row['%s' % blue_crit] == 'outflow') & (row['%s' % red_crit] == 'disk'):
            mapp_out_amp[int(row['Y']), int(row['X'])] = row[amp_blue]
            mapp_out_vel[int(row['Y']), int(row['X'])] = row[vel_blue]
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_blue] * 2.355)**2 - 100**2)
            mapp_disk_amp[int(row['Y']), int(row['X'])] = row[amp_red]
            mapp_disk_vel[int(row['Y']), int(row['X'])] = row[vel_red]
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_red] * 2.355)**2 - 100**2)  # fwhm - instrumental res fwhm
        elif (row['%s' % blue_crit] == 'disk') & (row['%s' % red_crit] == 'outflow'):
            mapp_out_amp[int(row['Y']), int(row['X'])] = row[amp_red]
            mapp_out_vel[int(row['Y']), int(row['X'])] = row[vel_red]
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_red] * 2.355)**2 - 100**2)
            mapp_disk_amp[int(row['Y']), int(row['X'])] = row[amp_blue]
            mapp_disk_vel[int(row['Y']), int(row['X'])] = row[vel_blue]
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_blue] * 2.355)**2 - 100**2)
        else:
            mapp_out_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.nan

    # blank out the edges using the original data
    mapp_out_amp[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_out_vel[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_out_fwhm[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_disk_amp[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_disk_vel[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_disk_fwhm[np.isnan(og_data[1])] = np.nan # [0] has some nans within

    # if we only want the NII/Halpha ratio maps....
    # or we only want to compare the different criteria....
    if (plot_NII_Halpha == True):
        return mapp_out_amp, mapp_out_vel, mapp_out_fwhm, mapp_disk_amp, mapp_disk_vel, mapp_disk_fwhm

    # otherwise....

    # now plot!
    fig = plt.figure(figsize=(10,15))
    fig.suptitle('%s (Criteria: %s)' % (line, criteria), fontsize=30)

    # first, plot the main criteria

    if criteria == 'NII_Halpha_Ratio':
        vmin = 0
        vmax = 5
    if criteria == 'Velocities':
        vmin = -300
        vmax = 300
    cmap = 'rainbow'

    ax = plt.subplot(4, 2, 1)
    im =  ax.imshow(outflow, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Outflow', fontsize=20)

    ax = plt.subplot(4, 2, 2)
    im =  ax.imshow(disk, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Disk', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)

    if criteria == 'NII_Halpha_Ratio':
        bar.set_label('[N II]/H-alpha', fontsize=18)
    elif criteria == 'Velocities':
        bar.set_label('velocity [km/s]', fontsize=18)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

    # amplitude
    vmin = 0
    vmax = 500

    ax = plt.subplot(4, 2, 3)
    im =  ax.imshow(mapp_out_amp, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

    ax = plt.subplot(4, 2, 4)
    im =  ax.imshow(mapp_disk_amp, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    bar = plt.colorbar(im, fraction=0.046)
    bar.set_label('flux', fontsize=18)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

    # velocity
    vmin = -300.
    vmax = 300.

    ax = plt.subplot(4, 2, 5)
    im =  ax.imshow(mapp_out_vel, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

    ax = plt.subplot(4, 2, 6)
    im =  ax.imshow(mapp_disk_vel, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    bar = plt.colorbar(im, fraction=0.046)
    bar.set_label('velocity [km/s]', fontsize=18)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

    # FWHM
    vmin = 0
    vmax = 300

    ax = plt.subplot(4, 2, 7)
    im =  ax.imshow(mapp_out_fwhm, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

    ax = plt.subplot(4, 2, 8)
    im =  ax.imshow(mapp_disk_fwhm, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    bar = plt.colorbar(im, fraction=0.046)
    bar.set_label('fwhm [km/s]', fontsize=18)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

    plt.savefig('%splots/fits2_%s_%s.png' % (savepath, line, criteria), dpi=200)

    return


def CHECK_NII_Halpha_Ratio2(og, infile, infile_err,
                            take_ratio=False, assign_comps=False, 
                            plot=False, savepath=False, line_to_plot=False):

    """
    This function tries to assign components based on the NII/Halpha ratio.
    """

    # read in the files
    infile = pd.read_csv(infile)
    infile_err = pd.read_csv(infile_err)

##############################################################################
# take ratio and propagate errors
##############################################################################

    if take_ratio == True:
        print('Calculating the [N II]/H-alpha ratio....')

        blue_rat = []
        red_rat = []
        blue_rat_err = []
        red_rat_err = []
        for index, row in tqdm(infile.iterrows()):

            x = row['X']
            y = row['Y']
            err = infile_err[(infile_err['X'] == x) & (infile_err['Y'] == y)]
            
            try:
                # calculate the ratio
                ratB = float(row['Amp5'])/float(row['Amp3'])
                ratB_err = Mult_Div_StdErr(float(row['Amp5']), float(row['Amp3']), 
                                           float(err['Amp5']), float(err['Amp3']))
            except:
                ratB = 'div0'
                ratB_err = 'div0'

            blue_rat.append(ratB)
            blue_rat_err.append(ratB_err)
                
            try:
                ratR = float(row['Amp6'])/float(row['Amp4'])
                ratR_err = Mult_Div_StdErr(float(row['Amp6']), float(row['Amp4']), 
                                        float(err['Amp6']), float(err['Amp4']))
            except:
                ratR = 'div0'
                ratR_err = 'div0'

            red_rat.append(ratR)
            red_rat_err.append(ratR_err)
    
        # print(infile)
        # infile = infile.drop(['BlueCompRatio', 'RedCompRatio']) # ugh
        infile['BlueCompRatio'] = blue_rat
        infile['RedCompRatio'] = red_rat

        infile_err['BlueCompRatio'] = blue_rat_err
        infile_err['RedCompRatio'] = red_rat_err

        infile.to_csv('%sfits2_NII_Halpha_Ratio.txt' % savepath, index=False)
        infile_err.to_csv('%sfits2_err_NII_Halpha_Ratio.txt' % savepath, index=False)


##############################################################################
# component assignment
##############################################################################
    
    if assign_comps == True:
        print('Assigning components based on ratio....')

        infile = pd.read_csv('%sfits2_NII_Halpha_Ratio.txt' % savepath)
        infile_err = pd.read_csv('%sfits2_NII_Halpha_Ratio.txt' % savepath)
        blue_rat = infile['BlueCompRatio']
        red_rat = infile['RedCompRatio']

        # take the median of the ratios to get a threshold
        # ratios = np.concatenate((blue_rat, red_rat))
        # ratio_thresh = np.median([float(i) for i in ratios if i != 'div0'])
        
        # assign components
        blue = []
        red = []
        for index, row in tqdm(infile.iterrows()):
            err_df = infile_err[(infile_err['X'] == row['X']) & (infile_err['Y'] == row['Y'])]  # grab errors

            if row['RedCompRatio'] == 'div0':
                red.append('und')
            else:
                if float(row['RedCompRatio']) > 2.:
                    red.append('outflow')
                elif float(row['RedCompRatio']) <= 2.:
                    red.append('disk')
                else:
                    red.append('und')

            if row['BlueCompRatio'] == 'div0':
                blue.append('und')
            else:
                if float(row['BlueCompRatio']) > 2.:
                    blue.append('outflow')
                elif float(row['BlueCompRatio']) <= 2.:
                    blue.append('disk')
                else:
                    blue.append('und')


            # if row['RedCompRatio'] == 'div0':
            #     red.append('und')
            # else:
            #     errR = float(err_df['RedCompRatio'])
            #     lineR = np.array([float(row['RedCompRatio']) - errR, 
            #                     float(row['RedCompRatio']) + errR])
            #     if (np.any(lineR > 2.)) and not (np.any(lineR <= 2.)):
            #         red.append('outflow')
            #     elif (np.any(lineR <= 2.)) and not (np.any(lineR > 2.)):
            #         print(lineR)
            #         red.append('disk')
            #     else:
            #         red.append('und')
                
            # if row['BlueCompRatio'] == 'div0':
            #     blue.append('und')
            # else:
            #     errB = float(err_df['BlueCompRatio'])
            #     lineB = np.array([float(row['BlueCompRatio']) - errB, 
            #                     float(row['BlueCompRatio']) + errB])
            #     if (np.any(lineB > 2.)) and not (np.any(lineB <= 2.)):
            #         blue.append('outflow')
            #     elif (np.any(lineB <= 2.)) and not (np.any(lineB > 2.)):
            #         blue.append('disk')
            #     else:
            #         blue.append('und')

        infile['CHECK_BLUE_NII_Halpha_Ratio'] = blue
        infile['CHECK_RED_NII_Halpha_Ratio'] = red
        infile.to_csv('%sfits2_NII_Halpha_Ratio.txt' % savepath, index=False)

    # option to plot
    if plot == True:

        # read in file
        infile = pd.read_csv('%sfits2_NII_Halpha_Ratio.txt' % savepath)

        # read in original data
        hdu = fits.open(og)[1]
        og_data = hdu.data
        y, x = og_data[1].shape

        # first, plot the ratios of the outflow vs the disk
        outflow = np.zeros((y,x))
        disk = np.zeros((y,x))
    
        for index, row in infile.iterrows():

            # account for the checks
            if ((row['CHECK1'] == 'FAIL') | (row['CHECK2'] == 'FAIL') | 
                (row['CHECK3'] == 'FAIL') | (row['CHECK4'] == 'FAIL') | 
                (row['CHECK_All_StoN'] == 'FAIL')):
                disk[int(row['Y']), int(row['X'])] = np.nan
                outflow[int(row['Y']), int(row['X'])] = np.nan
                continue

            # we are only keeping pixels where we have two defined component ratios
            # that we can compare. Otherwise, we say the pixel is undefined.
            if (row['CHECK_BLUE_NII_Halpha_Ratio'] == 'outflow') & (row['CHECK_RED_NII_Halpha_Ratio'] == 'disk'):
                outflow[int(row['Y']), int(row['X'])] = float(row['BlueCompRatio'])
                disk[int(row['Y']), int(row['X'])] = float(row['RedCompRatio'])
            elif (row['CHECK_BLUE_NII_Halpha_Ratio'] == 'disk') & (row['CHECK_RED_NII_Halpha_Ratio'] == 'outflow'):
                outflow[int(row['Y']), int(row['X'])] = float(row['RedCompRatio'])
                disk[int(row['Y']), int(row['X'])] = float(row['BlueCompRatio'])
            else:
                disk[int(row['Y']), int(row['X'])] = np.nan
                outflow[int(row['Y']), int(row['X'])] = np.nan

        # blank out the edges
        outflow[np.isnan(og_data[1])] = np.nan # [0] has some nans within
        disk[np.isnan(og_data[1])] = np.nan # [0] has some nans within

        # plot the component assignment!
        assign_comps_mapps(og, infile, outflow, disk, line=line_to_plot, criteria='NII_Halpha_Ratio',
                           savepath=savepath, plot_NII_Halpha=False)
    
    return


def CHECK_Velocities2(og, infile, diskmap, assign_comps, savepath, plot=False, line_to_plot=False, plot_NII_Halpha=False):

    """
    This function assigns components based on velocities of the disk.
    """

    if assign_comps == True:
        print('Assigning components based on velocities....')

        infile = pd.read_csv(infile)
        diskmap = fits.open(diskmap)
        diskmap = diskmap[0].data

        diskmap = diskmap[np.isfinite(diskmap)]
        print('(min., max. of disk) = (%s, %s) km/s.' % (np.round(np.min(diskmap),3), np.round(np.max(diskmap),3)))

        blue = []
        red = []
        for index, row in infile.iterrows():

            # use H-alpha velocities

            # make an array that spans the width of the line
            # due to instrumental resolution and the error on the line
            # TODO: add errors
            lineB = np.array([float(row['Vel3']) - 20, float(row['Vel3']) + 20])
            lineR = np.array([float(row['Vel4']) - 20, float(row['Vel4']) + 20])

            # Compare: if either the min or max falls within the disk map range of velocities,
            # call it the disk. We use min and max because the width of the line should be
            # less than the range of velocities

            # blueshifted component
            if (np.any(lineB > np.min(diskmap))) & (np.any(lineB < np.max(diskmap))):
                blue.append('disk')
            elif (np.any(lineB < np.min(diskmap))) | (np.any(lineB > np.max(diskmap))):
                blue.append('outflow')

            # redshifted component
            if (np.any(lineR > np.min(diskmap))) & (np.any(lineR < np.max(diskmap))):
                red.append('disk')
            elif (np.any(lineR < np.min(diskmap))) | (np.any(lineR > np.max(diskmap))):
                red.append('outflow')

        infile['CHECK_BLUE_Velocities'] = blue
        infile['CHECK_RED_Velocities'] = red

        infile.to_csv('%sfits2_Velocities.txt' % savepath, index=False)

    # option to plot
    if plot == True:
        
        # read in file
        infile = pd.read_csv('%sfits2_Velocities.txt' % savepath)

        # get info of original data
        hdu = fits.open(og)[1]
        og_data = hdu.data
        y, x = og_data[1].shape

        # first, plot the ratios of the outflow vs the disk
        outflow = np.zeros((y,x))
        disk = np.zeros((y,x))
    
        for index, row in infile.iterrows():

            # account for the checks
            if ((row['CHECK1'] == 'FAIL') | (row['CHECK2'] == 'FAIL') | 
                (row['CHECK3'] == 'FAIL') | (row['CHECK4'] == 'FAIL') | 
                (row['CHECK_All_StoN'] == 'FAIL')):
                disk[int(row['Y']), int(row['X'])] = np.nan
                outflow[int(row['Y']), int(row['X'])] = np.nan
                continue

            # if they are both the same outcome... undetermined!
            if row['CHECK_BLUE_Velocities'] == row['CHECK_RED_Velocities']:
                disk[int(row['Y']), int(row['X'])] = np.nan
                outflow[int(row['Y']), int(row['X'])] = np.nan

            # otherwise, assign them
            elif (row['CHECK_BLUE_Velocities'] == 'outflow') & (row['CHECK_RED_Velocities'] == 'disk'):
                outflow[int(row['Y']), int(row['X'])] = float(row['Vel3'])
                disk[int(row['Y']), int(row['X'])] = float(row['Vel4'])
            elif (row['CHECK_BLUE_Velocities'] == 'disk') & (row['CHECK_RED_Velocities'] == 'outflow'):
                outflow[int(row['Y']), int(row['X'])] = float(row['Vel4'])
                disk[int(row['Y']), int(row['X'])] = float(row['Vel3'])

        # blank out the edges
        outflow[np.isnan(og_data[1])] = np.nan # [0] has some nans within
        disk[np.isnan(og_data[1])] = np.nan # [0] has some nans within

        # plot the component assignment!
        assign_comps_mapps(og, infile, outflow, disk, line=line_to_plot, criteria='Velocities',
                           savepath=savepath, plot_NII_Halpha=False)
        
    if plot_NII_Halpha == True:

        print('Plotting NII/Halpha ratio based on velocity criteria....')
        np.seterr(divide='ignore', invalid='ignore')

        outflow = False
        disk = False
        infile = pd.read_csv('%sfits2_Velocities.txt' % savepath)

        # do the maps for both NIIb and Halpha
        NIIb_out_amp, _, _, NIIb_disk_amp, _, _ \
        = assign_comps_mapps(og, infile, outflow, disk, line='NIIb', criteria='Velocities',
                        savepath=savepath, plot_NII_Halpha=plot_NII_Halpha)
        
        Halpha_out_amp, _, _, Halpha_disk_amp, _, _ \
        = assign_comps_mapps(og, infile, outflow, disk, line='Halpha', criteria='Velocities',
                        savepath=savepath, plot_NII_Halpha=plot_NII_Halpha)

        vmin = 0
        vmax = 5
        cmap = 'rainbow'

        fig = plt.figure(figsize=(11,6))
        fig.suptitle('[NII/Halpha] (Criteria: Velocities)', fontsize=25)
        ax = plt.subplot(1, 2, 1)
        im =  ax.imshow(NIIb_out_amp / Halpha_out_amp, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        ax.set_title('Outflow', fontsize=20)

        ax = plt.subplot(1, 2, 2)
        im =  ax.imshow(NIIb_disk_amp / Halpha_disk_amp, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        ax.set_title('Disk', fontsize=20)
        bar = plt.colorbar(im, fraction=0.046)
        bar.set_label('[N II]/H-alpha', fontsize=18)
        bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

        plt.savefig('%splots/fits2_NII_Halpha_VELOCITIES.png' % savepath, dpi=200)

        out = NIIb_out_amp / Halpha_out_amp
        print('Mean outflow:', np.mean(out[np.isfinite(out)]))

        disk = NIIb_disk_amp / Halpha_disk_amp
        print('Mean disk:', np.mean(disk[np.isfinite(disk)]))

    return


def assign_comps_mapps_ALL(og, infile, line, criteria, savepath):

    # get info of original data
    hdu = fits.open(og)[1]
    og_data = hdu.data
    y, x = og_data[1].shape
    
    # use the original data to create the dimensions of the maps
    mapp_out_amp = np.zeros((y,x))
    mapp_disk_amp = np.zeros((y,x))
    mapp_out_vel = np.zeros((y,x))
    mapp_disk_vel = np.zeros((y,x))
    mapp_out_fwhm = np.zeros((y,x))
    mapp_disk_fwhm = np.zeros((y,x))

    if criteria == 'Velocities_only':
        criteria_out = 'outflow_VEL'
        criteria_disk = 'disk_VEL'
    elif criteria == 'Ratios_only':
        criteria_out = 'outflow_RATIO'
        criteria_disk = 'disk_RATIO'
    elif criteria == 'Velocities_and_Ratio':
        criteria_out = 'outflow_'
        criteria_disk = 'disk_'

    if line == 'Halpha':
        amp_blue = 'Amp3'
        vel_blue = 'Vel3'
        sig_blue = 'SigVel3'
        amp_red = 'Amp4'
        vel_red = 'Vel4'
        sig_red = 'SigVel4'
    
    if line == 'NIIb':
        amp_blue = 'Amp5'
        vel_blue = 'Vel5'
        sig_blue = 'SigVel5'
        amp_red = 'Amp6'
        vel_red = 'Vel6'
        sig_red = 'SigVel6'

    for index, row in infile.iterrows():

        # account for the checks
        # including signal to noise
        if ((row['CHECK1'] == 'FAIL') | (row['CHECK2'] == 'FAIL') | 
            (row['CHECK3'] == 'FAIL') | (row['CHECK4'] == 'FAIL') | 
            (row['CHECK_All_StoN'] == 'FAIL')):
            mapp_out_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.nan
            continue

        if (criteria_out in row['CHECK_BLUE_Velocities_Ratio']) & (criteria_disk in row['CHECK_RED_Velocities_Ratio']):
            mapp_out_amp[int(row['Y']), int(row['X'])] = row[amp_blue]
            mapp_out_vel[int(row['Y']), int(row['X'])] = row[vel_blue]
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_blue] * 2.355)**2 - 100**2)
            mapp_disk_amp[int(row['Y']), int(row['X'])] = row[amp_red]
            mapp_disk_vel[int(row['Y']), int(row['X'])] = row[vel_red]
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_red] * 2.355)**2 - 100**2)
        elif (criteria_disk in row['CHECK_BLUE_Velocities_Ratio']) & (criteria_out in row['CHECK_RED_Velocities_Ratio']):
            mapp_out_amp[int(row['Y']), int(row['X'])] = row[amp_red]
            mapp_out_vel[int(row['Y']), int(row['X'])] = row[vel_red]
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_red] * 2.355)**2 - 100**2)
            mapp_disk_amp[int(row['Y']), int(row['X'])] = row[amp_blue]
            mapp_disk_vel[int(row['Y']), int(row['X'])] = row[vel_blue]
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.sqrt((row[sig_blue] * 2.355)**2 - 100**2)
        else:
            mapp_out_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_out_fwhm[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_amp[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_vel[int(row['Y']), int(row['X'])] = np.nan
            mapp_disk_fwhm[int(row['Y']), int(row['X'])] = np.nan

    # blank out the edges using the original data
    mapp_out_amp[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_out_vel[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_out_fwhm[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_disk_amp[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_disk_vel[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    mapp_disk_fwhm[np.isnan(og_data[1])] = np.nan # [0] has some nans within


    fig = plt.figure(figsize=(10,15))
    fig.suptitle('%s (Criteria: %s)' % (line, criteria), fontsize=30)
    cmap = 'rainbow'

    # amp
    vmin = 0
    vmax = 500

    ax = plt.subplot(3, 2, 1)
    im =  ax.imshow(mapp_out_amp, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Outflow', fontsize=20)

    ax = plt.subplot(3, 2, 2)
    im =  ax.imshow(mapp_disk_amp, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Disk', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.set_label('flux', fontsize=18)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

    # vel
    vmin = -300
    vmax = 300

    ax = plt.subplot(3, 2, 3)
    im =  ax.imshow(mapp_out_vel, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Outflow', fontsize=20)

    ax = plt.subplot(3, 2, 4)
    im =  ax.imshow(mapp_disk_vel, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Disk', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.set_label('velocity [km/s]', fontsize=18)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

    # fwhm
    vmin = 0
    vmax = 300

    ax = plt.subplot(3, 2, 5)
    im =  ax.imshow(mapp_out_fwhm, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Outflow', fontsize=20)

    ax = plt.subplot(3, 2, 6)
    im =  ax.imshow(mapp_disk_fwhm, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_title('Disk', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.set_label('fwhm', fontsize=18)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')

    plt.savefig('%splots/fits2_%s_%s.png' % (savepath, line, criteria), dpi=200)

    return


def CHECK_Velocities_Ratio(og, infile, savepath, plot, line_to_plot):

    """
    This function mainly uses the velocities as a way to separate components.
    It then will try to fill in the gaps with the NII/Halpha ratio!
    We'll see how this goes.
    """

    print('Separating components....')
    
    # read in files
    infile = pd.read_csv(infile)
    ratio = pd.read_csv('%sfits2_NII_Halpha_Ratio.txt' % savepath)
    vels = pd.read_csv('%sfits2_Velocities.txt' % savepath)

    infile['BlueCompRatio'] = ratio['BlueCompRatio']
    infile['RedCompRatio'] = ratio['RedCompRatio']
    infile['CHECK_BLUE_Velocities'] = vels['CHECK_BLUE_Velocities']
    infile['CHECK_RED_Velocities'] = vels['CHECK_RED_Velocities']
    infile['CHECK_BLUE_NII_Halpha_Ratio'] = ratio['CHECK_BLUE_NII_Halpha_Ratio']
    infile['CHECK_RED_NII_Halpha_Ratio'] = ratio['CHECK_RED_NII_Halpha_Ratio']

    # read in original data
    hdu = fits.open(og)[1]
    og_data = hdu.data
    y, x = og_data[1].shape
    
    blue = []
    red = []

    for index, row in infile.iterrows():

        # account for the checks
        if ((row['CHECK1'] == 'FAIL') | (row['CHECK2'] == 'FAIL') | 
            (row['CHECK3'] == 'FAIL') | (row['CHECK4'] == 'FAIL') | 
            (row['CHECK_All_StoN'] == 'FAIL')):
            blue.append('und')
            red.append('und')
            continue

        # assign based on velocities; note where the determination comes from
        if (row['CHECK_BLUE_Velocities'] == 'outflow') & (row['CHECK_RED_Velocities'] == 'disk'):
            blue.append('outflow_VEL')
            red.append('disk_VEL')
        elif (row['CHECK_BLUE_Velocities'] == 'disk') & (row['CHECK_RED_Velocities'] == 'outflow'):
            red.append('outflow_VEL')
            blue.append('disk_VEL')

        # otherwise, try ratio!
        else:
            if (row['CHECK_BLUE_NII_Halpha_Ratio'] == 'outflow') & (row['CHECK_RED_NII_Halpha_Ratio'] == 'disk'):
                blue.append('outflow_RATIO')
                red.append('disk_RATIO')
            elif (row['CHECK_BLUE_NII_Halpha_Ratio'] == 'disk') & (row['CHECK_RED_NII_Halpha_Ratio'] == 'outflow'):
                red.append('outflow_RATIO')
                blue.append('disk_RATIO')
            else:
                blue.append('und')
                red.append('und')
    
    infile['CHECK_BLUE_Velocities_Ratio'] = blue
    infile['CHECK_RED_Velocities_Ratio'] = red
    infile.to_csv('%sfits2_Velocities_Ratio.txt' % savepath, index=False)

    if plot == True:

########################################################################
# first, use just the velocity criteria
########################################################################

        print('...based on velocities.')
        
        criteria = 'Velocities_only'
        assign_comps_mapps_ALL(og, infile, line_to_plot, criteria, savepath)

        
########################################################################
# second, use the velocity criteria and fill in the gaps with the ratio
########################################################################

        print('...based on velocities and the ratio.')

        criteria = 'Velocities_and_Ratio'
        assign_comps_mapps_ALL(og, infile, line_to_plot, criteria, savepath)

########################################################################
# finally, just look at where the ratios were used
########################################################################

        print('...based on only ratio.')
        criteria = 'Ratios_only'
        assign_comps_mapps_ALL(og, infile, line_to_plot, criteria, savepath)


#TODO: add check for broad component


