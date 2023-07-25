"""
Created on Tue June 27 2023

@author: Serena A. Cronin

This script is an example of reordering and assigning the components of a one Gaussian fit.
It will produce intensity maps of each the blueshifted and redshifted components.

"""

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from reorder1 import *
# from assign1 import *

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
# REORDER THE COMPONENTS BASED ON WAVELENGTH
##############################################################################

# define our rest wavelengths for the NII doublet, H-alpha, and the SII doublet
restwls = [6548.05, 6562.801, 6583.45, 6716.44, 6730.82]
Vsys = 243. # km/s
savepath = '../ngc253/June21/fits1_total/'
reorder = True
vels = True
errs = True
flux = True

if reorder == True:
    # reorder the components for both the parameter file and the error file
    reorder_components1(infile='%sfits1.txt' % savepath, outfile='%sfits1_reordered.txt' % savepath,
                    infile_err='%sfits1_err.txt' % savepath, outfile_err='%sfits1_err_reordered.txt' % savepath)

if vels == True:
    # add velocities to the parameter file
    add_velocities1(infile='%sfits1_reordered.txt' % savepath,
                    err_infile='%sfits1_err_reordered.txt' % savepath,
                    outfile='%sfits1_reordered.txt' % savepath, 
                    err_outfile='%sfits1_err_reordered.txt' % savepath,
                    restwls=restwls, Vsys=Vsys, i=78.)
    
if errs == True:
    # get the true errors by multiplying the errors by the rms per pixel
    true_errors(infile='%sfits1_reordered.txt' % savepath,
                    err_infile='%sfits1_err_reordered.txt' % savepath,
                    outfile='%sfits1_reordered.txt' % savepath, 
                    err_outfile='%sfits1_err_reordered.txt' % savepath)

# if flux == True:
#     # read in the reordered file
#     fits2 = pd.read_csv('%sfits1_reordered.txt' % savepath)

#     # produce intensity maps of H-alpha...
#     flux_map2(og='../ngc253/data/ADP.2018-11-22T21_29_46.157.fits',
#                 infile='%sfits1_reordered.txt' % savepath, 
#                 outfile1 = '%shalpha-fits1-flux-blue.fits' % savepath,
#                 outfile2='%shalpha-fits1-flux-red.fits' % savepath,
#                 line='Halpha')

#     # ...and the brighter NII line
#     flux_map2(og='../ngc253/data/ADP.2018-11-22T21_29_46.157.fits',
#                 infile='%sfits2_reordered.txt' % savepath, 
#                 outfile1 = '%sniib-fits2-flux-blue.fits' % savepath,
#                 outfile2='%sniib-fits2-flux-red.fits' % savepath,
#                 line='NIIb')

##############################################################################
# ASSIGN COMPONENTS TO EITHER THE DISK OR THE OUTFLOW
##############################################################################
# savepath = '../ngc253/June21/'

# check_fits = True
# sig_to_noise = True
# velocities = True
# ratios = True
# vels_rats = True

# plot_s2n = True
# plot_velocities = True
# take_ratio = True
# assign_comps = True
# plot_ratios = True
# plot_NII_Halpha = True
# plot_compare_crit = True
# line = 'Halpha'

# if check_fits == True:
#     physical_fit2(infile='%sfits2_reordered.txt' % savepath, savepath=savepath)

# if sig_to_noise == True:
#     SigToNoise(infile='%sfits2_reordered.txt' % savepath, 
#                infile_err='%sfits2_err_reordered.txt' % savepath,
#                savepath=savepath, plot=plot_s2n,
#                og='../ngc253/data/ADP.2018-11-22T21_29_46.157.fits')
    
# if velocities == True:
#     CHECK_Velocities2(og='../ngc253/data/ADP.2018-11-22T21_29_46.157.fits',
#                       infile='%sfits2_reordered_S2N.txt' % savepath,
#                       diskmap='../ngc253/data/ngc253_se_halpha_vel_model_smooth_FINAL.fits',
#                         assign_comps=assign_comps, savepath=savepath, plot=plot_velocities, 
#                         line_to_plot=line, plot_NII_Halpha=plot_NII_Halpha)

# if ratios == True:
#     CHECK_NII_Halpha_Ratio2(og='../ngc253/data/ADP.2018-11-22T21_29_46.157.fits',
#                             infile='%sfits2_reordered_S2N.txt' % savepath,
#                             infile_err='%sfits2_err_reordered.txt' % savepath,
#                             take_ratio=take_ratio, assign_comps=assign_comps,
#                             plot=plot_ratios, savepath=savepath, line_to_plot=line)
    
# if vels_rats == True:
#     CHECK_Velocities_Ratio(og='../ngc253/data/ADP.2018-11-22T21_29_46.157.fits',
#                             infile='%sfits2_reordered_S2N.txt' % savepath, 
#                             savepath=savepath, plot=plot_compare_crit, line_to_plot=line)