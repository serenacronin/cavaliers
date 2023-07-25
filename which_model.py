# imports
import sys
sys.path.append('../astro_tools')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
from matplotlib.offsetbox import AnchoredText

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

R = 2989
run_BIC = False
visual_BIC = True
dBIC_thresh = -50
dBIC_vmin = -200
dBIC_vmax = -10
visual_BIC_phys = True

def calc_BIC(infile, num_obs, free_params):

    fits = pd.read_csv(infile)
    DOF = num_obs - free_params  # number of observed points - free parameters
    chisq = fits['RedChiSq'] * DOF

    BIC = chisq + free_params*np.log(num_obs)

    fits['BIC'] = BIC

    fits.to_csv(infile, index=False)

    return fits


def physical_check(infile, i, j, fitnum):
    print(i,j, end='\r')
    fits = pd.read_csv(infile)
    fits = fits[(fits['Y'] == i) & (fits['X'] == j)]

    if fits.empty == True:
        return 'FAIL'

    if fitnum == 1:
            if (float(fits['Sig2'])*2.355 < (6562.801 / R)):
                return 'FAIL'

            # is FWHM greater than ~2 * 450 km/s? (absolute value)
            elif (float(fits['SigVel2']) > 1000):
                return 'FAIL'

            # are the velocities unphysical?
            # i.e., greater than ~600 km/s (absolute value)
            elif (np.abs(float(fits['Vel2'])) > 600):
                return 'FAIL'
            else:
                return 'PASS'

    if fitnum == 2:
        if (float(fits['Sig3'])*2.355 < (6562.801 / R)) | (float(fits['Sig4'])*2.355 < (6562.801 / R)):
            return 'FAIL'

        # is FWHM greater than ~2 * 450 km/s? (absolute value)
        elif ((float(fits['SigVel3']) > 1000) | (float(fits['SigVel4']) > 1000)):
            return 'FAIL'

        # are the velocities similar between components?
        # arbitrarily choosing less than 1/2 resolution
        elif (np.abs(float(fits['Wvl3']) - float(fits['Wvl4'])) < (6562.80 / R)/2):
            return 'FAIL'

        # are the velocities unphysical?
        # i.e., greater than ~600 km/s (absolute value)
        elif (np.abs(float(fits['Vel3'])) > 600) | (np.abs(float(fits['Vel4'])) > 600):
            return 'FAIL'
        else:
            return 'PASS'
        
    if fitnum == 3:
        if (float(fits['Sig4'])*2.355 < (6562.801 / R)) | (float(fits['Sig5'])*2.355 < (6562.801 / R)) | (float(fits['Sig6'])*2.355 < (6562.801 / R)):
            return 'FAIL'

        # is FWHM greater than ~2 * 450 km/s? (absolute value)
        elif (float(fits['SigVel4']) > 1000) | (float(fits['SigVel5']) > 1000) | (float(fits['SigVel6']) > 1000):
            return 'FAIL'

        # are the velocities similar between components?
        # arbitrarily choosing less than 1/2 resolution
        elif ((np.abs(float(fits['Wvl4']) - float(fits['Wvl5'])) < (6562.80 / R)/2) | (np.abs(float(fits['Wvl4']) - float(fits['Wvl6'])) < (6562.80 / R)/2) | 
                (np.abs(float(fits['Wvl5']) - float(fits['Wvl6'])) < (6562.80 / R)/2)):
            return 'FAIL'

        # are the velocities unphysical?
        # i.e., greater than ~600 km/s (absolute value)
        elif (np.abs(float(fits['Vel4'])) > 600) | (np.abs(float(fits['Vel5'])) > 600) | (np.abs(float(fits['Vel6'])) > 600):
            return 'FAIL'
        else:
            return 'PASS'


# =====================================================================
# Calculate the BICs
# =====================================================================

num_obs = 150
free_params1 = 6
free_params2 = 12
free_params3 = 18
savepath1 = '../ngc253/June21/fits1_total/'
savepath2 = '../ngc253/June21/fits2_total/'
savepath3 = '../ngc253/June21/fits3_total/'
infile1 = '%sfits1_reordered.txt' % savepath1
infile2 = '%sfits2_reordered.txt' % savepath2
infile3 = '%sfits3_reordered.txt' % savepath3

if run_BIC == True:
    print('Calculating the BIC values....')
    fits1 = calc_BIC(infile1, num_obs, free_params1)
    fits2 = calc_BIC(infile2, num_obs, free_params2)
    fits3 = calc_BIC(infile3, num_obs, free_params3)
else:
    fits1 = pd.read_csv(infile1)
    fits2 = pd.read_csv(infile2)
    fits3 = pd.read_csv(infile3)

# =====================================================================
# Compare the BICs visually
# =====================================================================

if visual_BIC == True:
    print('Plotting the BIC values....')
    savepath = '../ngc253/June21/'
    og = '../ngc253/data/ADP.2018-11-22T21_29_46.157.fits'

    # print(np.mean(fits1['BIC']), np.median(fits1['BIC']), np.min(fits1['BIC']), np.max(fits1['BIC']))
    # print(np.mean(fits2['BIC']), np.median(fits2['BIC']), np.min(fits2['BIC']), np.max(fits2['BIC']))
    # print(np.mean(fits3['BIC']), np.median(fits3['BIC']), np.min(fits3['BIC']), np.max(fits3['BIC']))

    # get info of original data
    hdu = fits.open(og)[1]
    og_data = hdu.data
    y, x = og_data[1].shape

    # use the original data to create the dimensions of the maps
    BIC_map1 = np.empty((y,x))
    BIC_map2 = np.empty((y,x))
    BIC_map3 = np.empty((y,x))
    redchisq_map1 = np.empty((y,x))
    redchisq_map2 = np.empty((y,x))
    redchisq_map3 = np.empty((y,x))

    # make maps of the BICs
    for index, row in fits1.iterrows():
        redchisq_map1[int(row['Y']), int(row['X'])] = row['RedChiSq']
        BIC_map1[int(row['Y']), int(row['X'])] = row['BIC']
    for index, row in fits2.iterrows():
        redchisq_map2[int(row['Y']), int(row['X'])] = row['RedChiSq']
        BIC_map2[int(row['Y']), int(row['X'])] = row['BIC']
    for index, row in fits3.iterrows():
        redchisq_map3[int(row['Y']), int(row['X'])] = row['RedChiSq']
        BIC_map3[int(row['Y']), int(row['X'])] = row['BIC']

    # blank out edges
    BIC_map1[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    BIC_map2[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    BIC_map3[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    redchisq_map1[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    redchisq_map2[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    redchisq_map3[np.isnan(og_data[1])] = np.nan # [0] has some nans within

    plt.figure(figsize=(14,14))
    ax = plt.subplot(3, 3, 1)
    im = ax.imshow(BIC_map1, origin='lower', vmin=50., vmax=1000., cmap='plasma_r')
    ax.set_title('BIC one system', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])
    at = AnchoredText(
    'Mean: %s' % int(round(np.mean((fits1['BIC'])),2)), prop=dict(size=18), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    ax = plt.subplot(3, 3, 2)
    im = ax.imshow(BIC_map2, origin='lower', vmin=50., vmax=1000., cmap='plasma_r')
    ax.set_title('BIC two system', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])
    at = AnchoredText(
    'Mean: %s' % int(round(np.mean((fits2['BIC'])),2)), prop=dict(size=18), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    ax = plt.subplot(3, 3, 3)
    im = ax.imshow(BIC_map3, origin='lower', vmin=50., vmax=1000., cmap='plasma_r')
    ax.set_title('BIC three system', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])
    at = AnchoredText(
    'Mean: %s' % int(round(np.mean((fits3['BIC'])),2)), prop=dict(size=18), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    ax = plt.subplot(3, 3, 4)
    im = ax.imshow(BIC_map2 - BIC_map1, origin='lower', vmin=dBIC_vmin, vmax=dBIC_vmax, cmap='plasma_r')
    ax.set_title('$\Delta BIC =$ BIC$_{2}$ - BIC$_{1}$', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(3, 3, 5)
    im = ax.imshow(BIC_map3 - BIC_map1, origin='lower', vmin=dBIC_vmin, vmax=dBIC_vmax, cmap='plasma_r')
    ax.set_title('$\Delta BIC =$ BIC$_{3}$ - BIC$_{1}$', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(3, 3, 6)
    im = ax.imshow(BIC_map3 - BIC_map2, origin='lower', vmin=dBIC_vmin, vmax=dBIC_vmax, cmap='plasma_r')
    ax.set_title('$\Delta BIC =$ BIC$_{3}$ - BIC$_{2}$', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(3, 3, 7)
    im = ax.imshow(redchisq_map1, origin='lower', vmin=0., vmax=10., cmap='plasma_r')
    ax.set_title('$\chi^{2}_{r}$ one system', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(3, 3, 8)
    im = ax.imshow(redchisq_map2, origin='lower', vmin=0., vmax=10., cmap='plasma_r')
    ax.set_title('$\chi^{2}_{r}$ two system', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(3, 3, 9)
    im = ax.imshow(redchisq_map3, origin='lower', vmin=0., vmax=10., cmap='plasma_r')
    ax.set_title('$\chi^{2}_{r}$ three system', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('%sBIC_vs_RedChiSq_zoom.png' % savepath, dpi=200)

    # now create a map based on comparison
    BIC_map_total = np.empty((y,x))

    for i in np.arange(y):
        for j in np.arange(x):
            BIC_1 = BIC_map1[i,j]
            BIC_2 = BIC_map2[i,j]
            BIC_3 = BIC_map3[i,j]
            # print(BIC_1, BIC_2, BIC_3)

            if (BIC_2 - BIC_1) < dBIC_thresh:  # if 2 is better than 1
                if (BIC_3 - BIC_2) < dBIC_thresh:  # but 3 is better than 2
                    BIC_map_total[i,j] = 3  # then 3 is the best!
                else:
                    BIC_map_total[i,j] = 2  # otherwise, 2 is the best!
            elif (BIC_3 - BIC_1) < dBIC_thresh:  # or, if 3 is better than 1
                if (BIC_2 - BIC_3) < dBIC_thresh:  # but 2 is better than 3
                    BIC_map_total[i,j] = 2  # then 2 is the best!
                else:
                    BIC_map_total[i,j] = 3  # otherwise, 3 is the best!
            else:
                BIC_map_total[i,j] = 1  # OTHERWISE...1 wins!

    # print(BIC_map1[268,262], BIC_map2[268,262], BIC_map3[268,262], BIC_map_total[268,262])

    # blank out the edges
    BIC_map_total[np.isnan(og_data[1])] = np.nan # [0] has some nans within

    plt.figure(figsize=(7,7))
    ax = plt.subplot(1, 1, 1)
    im = ax.imshow(BIC_map_total, origin='lower', vmin=1, vmax=3, cmap='cool')
    ax.set_title('BIC Comparison', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()
    plt.close()


if visual_BIC_phys == True:

    savepath = '../ngc253/June21/'
    og = '../ngc253/data/ADP.2018-11-22T21_29_46.157.fits'
    
    # get info of original data
    hdu = fits.open(og)[1]
    og_data = hdu.data
    y, x = og_data[1].shape

    # use the original data to create the dimensions of the maps
    BIC_map1 = np.empty((y,x))
    BIC_map2 = np.empty((y,x))
    BIC_map3 = np.empty((y,x))
    redchisq_map1 = np.empty((y,x))
    redchisq_map2 = np.empty((y,x))
    redchisq_map3 = np.empty((y,x))

    # make maps of the BICs
    for index, row in fits1.iterrows():
        redchisq_map1[int(row['Y']), int(row['X'])] = row['RedChiSq']
        BIC_map1[int(row['Y']), int(row['X'])] = row['BIC']
    for index, row in fits2.iterrows():
        redchisq_map2[int(row['Y']), int(row['X'])] = row['RedChiSq']
        BIC_map2[int(row['Y']), int(row['X'])] = row['BIC']
    for index, row in fits3.iterrows():
        redchisq_map3[int(row['Y']), int(row['X'])] = row['RedChiSq']
        BIC_map3[int(row['Y']), int(row['X'])] = row['BIC']

    # blank out edges
    BIC_map1[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    BIC_map2[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    BIC_map3[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    redchisq_map1[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    redchisq_map2[np.isnan(og_data[1])] = np.nan # [0] has some nans within
    redchisq_map3[np.isnan(og_data[1])] = np.nan # [0] has some nans within

    # compare the BICs AND see if the fits are physical
    BIC_map_total = np.empty((y,x))
    for i in np.arange(y):
        for j in np.arange(x):
            BIC_1 = BIC_map1[i,j]
            BIC_2 = BIC_map2[i,j]
            BIC_3 = BIC_map3[i,j]

            # FIXME: what if 2 is nan, but 1 and 3 aren't? or vice versa
            if ((np.isfinite(BIC_1) == False) | (np.isfinite(BIC_2) == False) |
                (np.isfinite(BIC_3) == False)):
                BIC_map_total[i,j] = np.nan
                continue
            
            elif (BIC_2 - BIC_1) < dBIC_thresh:  # if 2 is better than 1
                if (BIC_3 - BIC_2) < dBIC_thresh:  # but 3 is better than 2
                    # check physicality of 3
                    result = physical_check(infile=infile3, i=i, j=j, fitnum=3)
                    if result == 'PASS':  # if 3 is physical...
                        BIC_map_total[i,j] = 3  # ...then 3 is the best!
                    else:
                        pass
                else:
                    # check physicality of 2
                    result = physical_check(infile=infile2, i=i, j=j, fitnum=2)
                    if result == 'PASS':  # if 2 is physical...
                        BIC_map_total[i,j] = 2  # ...then 2 is the best!
                    else:
                        pass
            elif (BIC_3 - BIC_1) < dBIC_thresh:  # if 3 is better than 1
                if (BIC_2 - BIC_3) < dBIC_thresh:  # but 2 is better than 3
                    # check physicality of 2
                    result = physical_check(infile=infile2, i=i, j=j, fitnum=2)
                    if result == 'PASS':
                        BIC_map_total[i,j] = 2  # then 2 is the best!
                    else:
                        pass
                else:
                    # check the physicality of 3
                    result = physical_check(infile=infile3, i=i, j=j, fitnum=3)
                    if result == 'PASS':
                        BIC_map_total[i,j] = 3  # then 3 is the best!
                    else:
                        pass
            else:
                # check the physicality of 1
                result = physical_check(infile=infile1, i=i, j=j, fitnum=1)
                if result == 'PASS':
                    BIC_map_total[i,j] = 1  # OTHERWISE...1 wins!
                else:
                    BIC_map_total[i,j] = np.nan

    # blank out the edges
    BIC_map_total[np.isnan(og_data[1])] = np.nan # [0] has some nans within

    plt.figure(figsize=(7,7))
    ax = plt.subplot(1, 1, 1)
    im = ax.imshow(BIC_map_total, origin='lower', vmin=1, vmax=3, cmap='cool')
    ax.set_title('BIC Comparison + Physical Test', fontsize=20)
    bar = plt.colorbar(im, fraction=0.046)
    bar.ax.tick_params(width=2.5, labelsize=16, length=7, direction='in')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    plt.close()