#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:32:57 2022

@author: serenac
"""
import warnings
from astropy.io import fits
import astropy.io.fits as pyfits
import numpy as np
from astropy.wcs import wcs
from reproject import reproject_interp
import pandas as pd
import astropy.units as u
from astropy.coordinates import SpectralCoord
import aplpy
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
pd.options.display.max_columns = 99
warnings.filterwarnings("ignore")

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def vel_diff(files, line):
    """ 
    plot the difference in velocities 
    between the disk and the outflow per line
    """
    
    # get the header of one of the files
    get_hdr = pyfits.open(files[0])
    
    # read in the data
    outflow = pyfits.getdata(files[0])
    disk = pyfits.getdata(files[1])
    
    # subtract the outflow from the disk and make new fits hdu
    diff = disk - outflow
    w = wcs.WCS(get_hdr[0].header,naxis=2).celestial
    new_header = w.to_header()
    diff_fits = fits.PrimaryHDU(data=diff,header=new_header)

    # plot!
    gc = aplpy.FITSFigure(diff_fits)
    gc.show_colorscale(vmid=0,cmap='RdBu_r')
    gc.add_colorbar()
    gc.colorbar.set_axis_label_text(axis_label_text='$\Delta$ v [km/s]')
    gc.colorbar.set_font(size=18)
    gc.colorbar.set_axis_label_font(size=22)
    gc.colorbar.set_axis_label_rotation(-90)
    gc.colorbar.set_axis_label_pad(30)
    gc.axis_labels.set_font(size=18)
    gc.tick_labels.set_font(size=18)
    gc.set_title('%s: Disk - Outflow' % line, fontsize=22)
    
    return

def II_ratio(iia, iib, feature, line):
    
    # get the header of one of the files
    get_hdr = pyfits.open(iib)
    w = wcs.WCS(get_hdr[0].header,naxis=2).celestial
    
    # read in the data
    iia_data = pyfits.getdata(iia)
    iib_data = pyfits.getdata(iib)
    
    alphas = NormalizeData(iib_data)
    ratio = iib_data / iia_data
    
    # weights = (niib_data + niia_data) / max((niia_data+niib_data).flatten())
    # weighted_ratio = ratio*weights
    
    w = wcs.WCS(get_hdr[0].header,naxis=2).celestial
   # new_header = w.to_header()
   # ratio_fits = fits.PrimaryHDU(data=ratio,header=new_header)
      
    plt.subplot(projection=w)
    plt.imshow(ratio, cmap='gist_rainbow',alpha=alphas)
    cbar = plt.colorbar()
    
    # if (feature == 'Disk'):
        # plt.clim(0, 15)
        
    plt.clim(0,2)
        
    cbar.set_label(label='flux ratio',rotation=-90,labelpad=15,fontsize=12)
    plt.xlabel('RA (ICRS)', fontsize=14)
    plt.ylabel('Dec (ICRS)', fontsize=14)
    plt.title(' %s (%s)' % (line, feature), fontsize=14)
    plt.show()
    plt.close()
    
    return

def II_Ha_ratio(ii, ha, which_ii, feature, propty):
    
    # get the header of one of the files
    get_hdr = pyfits.open(ha)
    
    # read in the data
    ii_data = pyfits.getdata(ii)
    ha_data = pyfits.getdata(ha)
    
    # subtract the outflow from the disk and make new fits hdu
    ratio = ii_data / ha_data
    alphas = NormalizeData(ii_data)
        
    w = wcs.WCS(get_hdr[0].header,naxis=2).celestial
    #new_header = w.to_header()
    #ratio_fits = fits.PrimaryHDU(data=ratio,header=new_header)
    
    plt.subplot(projection=w)
    plt.imshow(ratio, cmap='gist_rainbow',alpha=alphas)
    cbar = plt.colorbar()
    
    # if (feature == 'Outflow') and (propty == 'flux') and (which_nii == 'a'):
    #     plt.clim(0, 5)
    # elif (feature == 'Outflow') and (propty == 'flux') and (which_nii == 'b'):
    #     plt.clim(0, 15)
    
    if (feature == 'Outflow') and (propty == 'flux'):
        plt.clim(0,5)
    else:
        plt.clim(0,2)
    
    cbar.set_label(label='%s ratio' % (propty),rotation=-90,labelpad=15,fontsize=12)
    plt.xlabel('RA (ICRS)', fontsize=14)
    plt.ylabel('Dec (ICRS)', fontsize=14)
    plt.title('%s / Ha (%s)' % (which_ii,feature),fontsize=14)
    plt.show()
    plt.close()
    return


if __name__ == '__main__':
    
    dont_plot_old = True
    
    if dont_plot_old == False:
        # plot velocity differences
        Ha = ['subcube_outflow_vel_Ha.fits','subcube_disk_vel_Ha.fits']
        NIIa = ['subcube_outflow_vel_NIIa.fits','subcube_disk_vel_NIIa.fits']
        NIIb = ['subcube_outflow_vel_NIIb.fits','subcube_disk_vel_NIIb.fits']
        vel_diff(Ha, 'Ha')
        vel_diff(NIIa, 'NIIa')
        vel_diff(NIIb, 'NIIb')
        
        # plot NIIb/NIIa (flux)
        NIIa = ['subcube_outflow_flux_NIIa.fits','subcube_disk_flux_NIIa.fits']
        NIIb = ['subcube_outflow_flux_NIIb.fits','subcube_disk_flux_NIIb.fits']
        NII_ratio(NIIa[0],NIIb[0],feature='Outflow')
        NII_ratio(NIIa[1],NIIb[1],feature='Disk')
        
        # plot NII/Ha
        NIIa = ['subcube_outflow_flux_NIIa.fits','subcube_disk_flux_NIIa.fits',
                'subcube_outflow_vel_NIIa.fits','subcube_disk_vel_NIIa.fits',
                'subcube_outflow_fwhm_NIIa.fits','subcube_disk_fwhm_NIIa.fits']
        NIIb = ['subcube_outflow_flux_NIIb.fits','subcube_disk_flux_NIIb.fits',
                'subcube_outflow_vel_NIIb.fits','subcube_disk_vel_NIIb.fits',
                'subcube_outflow_fwhm_NIIb.fits','subcube_disk_fwhm_NIIb.fits']
        Ha = ['subcube_outflow_flux_Ha.fits','subcube_disk_flux_Ha.fits',
                'subcube_outflow_vel_Ha.fits','subcube_disk_vel_Ha.fits',
                'subcube_outflow_fwhm_Ha.fits','subcube_disk_fwhm_Ha.fits']
        
        NII_Ha_ratio(NIIa[0], Ha[0], which_nii='a', feature='Outflow', propty='flux')
        NII_Ha_ratio(NIIa[1], Ha[1], which_nii='a', feature='Disk', propty='flux')
        NII_Ha_ratio(NIIb[0], Ha[0], which_nii='b', feature='Outflow', propty='flux')
        NII_Ha_ratio(NIIb[1], Ha[1], which_nii='b', feature='Disk', propty='flux')
    
    
        NII_Ha_ratio(NIIa[4], Ha[4], which_nii='a', feature='Outflow', propty='fwhm')
        NII_Ha_ratio(NIIa[5], Ha[5], which_nii='a', feature='Disk', propty='fwhm')
        NII_Ha_ratio(NIIb[4], Ha[4], which_nii='b', feature='Outflow', propty='fwhm')
        NII_Ha_ratio(NIIb[5], Ha[5], which_nii='b', feature='Disk', propty='fwhm')
        
        SIIa = ['subcube_outflow_flux_SIIa.fits','subcube_disk_flux_SIIa.fits']
        SIIb = ['subcube_outflow_flux_SIIb.fits','subcube_disk_flux_SIIb.fits']
        II_ratio(SIIa[0],SIIb[0],feature='Outflow', line='SIIb / SIIa')
        II_ratio(SIIa[1],SIIb[1],feature='Disk', line='SIIb / SIIa')
        
    else:
        
        # plot SII/Ha
        SIIa = ['subcube_outflow_flux_SIIa.fits','subcube_disk_flux_SIIa.fits',
                'subcube_outflow_vel_SIIa.fits','subcube_disk_vel_SIIa.fits',
                'subcube_outflow_fwhm_SIIa.fits','subcube_disk_fwhm_SIIa.fits']
        SIIb = ['subcube_outflow_flux_SIIb.fits','subcube_disk_flux_SIIb.fits',
                'subcube_outflow_vel_SIIb.fits','subcube_disk_vel_SIIb.fits',
                'subcube_outflow_fwhm_SIIb.fits','subcube_disk_fwhm_SIIb.fits']
        Ha = ['subcube_outflow_flux_Ha.fits','subcube_disk_flux_Ha.fits',
                'subcube_outflow_vel_Ha.fits','subcube_disk_vel_Ha.fits',
                'subcube_outflow_fwhm_Ha.fits','subcube_disk_fwhm_Ha.fits']
        
        II_Ha_ratio(SIIa[0], Ha[0], which_ii='SIIa', feature='Outflow', propty='flux')
        II_Ha_ratio(SIIa[1], Ha[1], which_ii='SIIa', feature='Disk', propty='flux')
        II_Ha_ratio(SIIb[0], Ha[0], which_ii='SIIb', feature='Outflow', propty='flux')
        II_Ha_ratio(SIIb[1], Ha[1], which_ii='SIIb', feature='Disk', propty='flux')
    
        II_Ha_ratio(SIIa[4], Ha[4], which_ii='SIIa', feature='Outflow', propty='fwhm')
        II_Ha_ratio(SIIa[5], Ha[5], which_ii='SIIa', feature='Disk', propty='fwhm')
        II_Ha_ratio(SIIb[4], Ha[4], which_ii='SIIb', feature='Outflow', propty='fwhm')
        II_Ha_ratio(SIIb[5], Ha[5], which_ii='SIIb', feature='Disk', propty='fwhm')