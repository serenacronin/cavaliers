#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:43:09 2022

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
from tqdm import tqdm
pd.options.display.max_columns = 99
warnings.filterwarnings("ignore")


def stitch(image_list, outfile):
    
    """
    Note: right now, this requires the cube to have been split 4 ways
    """

    # get the data into numpy arrays
    im1 = fits.getdata(image_list[0])
    im2 = fits.getdata(image_list[1])
    im3 = fits.getdata(image_list[2])
    im4 = fits.getdata(image_list[3])
    head = fits.getheader(image_list[0])  # grab the header

    # get the shape of the data
    nz1, ny1, nx1 = im1.shape
    nz2, ny2, nx2 = im2.shape
    nz3, ny3, nx3 = im3.shape
    nz4, ny4, nx4 = im4.shape

    # stitch them together into one ndarray
    result = np.zeros((nz1, int(ny1+ny2+ny3+ny4), nx1))
    
    result[:, :ny1, :] = im1
    result[:, ny1:(ny1+ny2), :] = im2
    result[:, (ny1+ny2):(ny1+ny2+ny3), :] = im3
    result[:, (ny1+ny2+ny3):, :] = im4

    # write to file!
    fits.writeto(outfile, result, header=head, overwrite=True)
    
    print('Cube stitched together!')
    return


def reproject_velcube(mycube, velcube, mycube_name, velcube_name):
    
    """
    This will reproject the model velocity cube into the footprint of our cube
    """

    # first, we need mycube to just have 2 axes
    # gimmie them ras and decs
    w = wcs.WCS(mycube[0].header, naxis=2).celestial
    new_header = w.to_header()
    new_cube = pyfits.PrimaryHDU(data=mycube[0].data[0, :, :],
                                 header=new_header)
    new_cube.writeto('%s_naxis2.fits' % mycube_name, overwrite=True)

    # awesome. now...open it and reproject to the footprint of our cube!
    subcube = pyfits.open('%s_naxis2.fits' % mycube_name)
    array, footprint = reproject_interp(velcube, subcube[0].header)
    pyfits.writeto('%s_reproject.fits' % velcube_name, array,
                   subcube[0].header, overwrite=True)
    
    print('Velocity model has been reprojected into the footprint of my cube!')
    return


def BigTable(mycube, mycube_name):
    
    """
    This will make a big table of the information that we want.
    Note: right now this is very specific to this project.
    
    """
    
    XX,YY = np.meshgrid(np.arange(mycube[0].shape[1]), 
                        np.arange(mycube[0].shape[0]))
    
    table = np.vstack((mycube[0].ravel(),
                       mycube[1].ravel(),
                       mycube[2].ravel(),
                       mycube[3].ravel(),
                       mycube[4].ravel(),
                       mycube[5].ravel(),
                       mycube[6].ravel(),
                       mycube[7].ravel(),
                       mycube[8].ravel(),
                       mycube[9].ravel(),
                       mycube[10].ravel(),
                       mycube[11].ravel(),
                       mycube[12].ravel(),
                       mycube[13].ravel(),
                       mycube[14].ravel(),
                       mycube[15].ravel(),
                       mycube[16].ravel(),
                       mycube[17].ravel(),
                       XX.ravel(),YY.ravel())).T
    
    df = pd.DataFrame(table)
    
    df.columns=['NIIa_Amp1', 'NIIa_Cen1', 'NIIa_Width1',
                'NIIa_Amp2', 'NIIa_Cen2', 'NIIa_Width2',
                'Ha_Amp1', 'Ha_Cen1', 'Ha_Width1',
                'Ha_Amp2', 'Ha_Cen2', 'Ha_Width2',
                'NIIb_Amp1', 'NIIb_Cen1', 'NIIb_Width1',
                'NIIb_Amp2', 'NIIb_Cen2', 'NIIb_Width2',
                'x', 'y']
    

    df.to_csv('%s_table.csv' % mycube_name, index=False)
    
    print('Initial table completed!')
    
    return


def separate_components(info, restfreq):
    
    """
    This will separate the disk and outflow components of each line.
    """
    
    peak1_result = []
    peak2_result = []
    cen1_vels = []
    cen2_vels = []
    
    for index, row in tqdm(info.iterrows(), total=info.shape[0]):
        
        wl_cen1 = SpectralCoord(row['Cen1'], unit='AA')
        wl_cen2 = SpectralCoord(row['Cen2'], unit='AA')
        
        vel_cen1 = (wl_cen1.to(u.km / u.s, doppler_rest = restfreq * u.AA,
                                                      doppler_convention='optical'))
        vel_cen2 = (wl_cen2.to(u.km / u.s, doppler_rest = restfreq * u.AA,
                                                      doppler_convention='optical'))
        
        peak1 = abs(row['vel_model'] - (vel_cen1).value)
        peak2 = abs(row['vel_model'] - (vel_cen2).value)
        
        cen1_vels.append(vel_cen1.value)
        cen2_vels.append(vel_cen2.value)
        
        if np.isfinite(row['vel_model']) == False:
            peak1_result.append('model NaN')
            peak2_result.append('model NaN')
            
        elif np.isfinite(row['vel_model']) != False:
            if peak1 < peak2:
                peak1_result.append('disk')
                peak2_result.append('outflow')
    
            elif peak1 > peak2:
                peak1_result.append('outflow')
                peak2_result.append('disk')
                
            elif peak1 == peak2:
                peak1_result.append('disk')  # iffy
                peak2_result.append('disk')  # iffy
                
            elif (np.isfinite(peak1) or np.isfinite(peak2)) == False:
                peak1_result.append('nan')
                peak2_result.append('nan')
      
    info['Cen1_Vel'] = cen1_vels
    info['Cen2_Vel'] = cen2_vels
    info['peak1'] = peak1_result
    info['peak2'] = peak2_result
    
    print('Components are separated!')
    return(info)


def mapp(x, y, myfits, info, line, feature):
    """ info: information specific to a line via DataFrame
        myfits: fits cube (for header)
        line: line we want to create the map for
        feature: outflow or disk of galaxy
        x: x-axis of cube/region
        y: y-axis of cube/region
    """
    
    # print(feature)
    # if (feature != 'outflow') or (feature !='disk'):
    #     raise ValueError('not a valid feature')
        
    # else:
    #     pass
    
    flux_mapp = np.zeros((y,x))
    vel_mapp  = np.zeros((y,x))
    fwhm_mapp = np.zeros((y,x))
    
    for index, row in tqdm(info.iterrows(), total=info.shape[0]):
        x = int(row['x_pix'])
        y = int(row['y_pix'])
    
        if row['peak1'] == feature:
            flux_mapp[y,x]  = row['Amp1']
            vel_mapp[y,x]   = row['Cen1_Vel']
            fwhm_mapp[y,x]  = row['Width1']
            
        elif row['peak2'] == feature:
            flux_mapp[y,x] = row['Amp2']
            vel_mapp[y,x]  = row['Cen2_Vel']
            fwhm_mapp[y,x]  = row['Width2']
            
        else:
            pass
    
    print('Maps are made...')
        
    # write maps to file
    w = wcs.WCS(myfits[0].header,naxis=2).celestial
    new_header = w.to_header()

    # flux
    outflow_flux = fits.PrimaryHDU(data=flux_mapp,header=new_header)
    outflow_flux.writeto('%s_flux_%s.fits' % (feature,line),overwrite=True)
    
    # velocity
    outflow_vel = fits.PrimaryHDU(data=vel_mapp,header=new_header)
    outflow_vel.writeto('%s_vel_%s.fits' % (feature,line),overwrite=True)
    
    # fwhm
    outflow_fwhm = fits.PrimaryHDU(data=fwhm_mapp,header=new_header)
    outflow_fwhm.writeto('%s_fwhm_%s.fits' % (feature,line),overwrite=True)
        
    print('...and written to file!')
    return

def line_df(line, restfreq, fit_info):
    
    line_df = pd.DataFrame()
    line_df['x_pix'] = fit_info['x']
    line_df['y_pix'] = fit_info['y']
    line_df['Amp1'] = fit_info['%s_Amp1' % line]
    line_df['Cen1'] = fit_info['%s_Cen1' % line]
    line_df['Width1'] = fit_info['%s_Width1' % line]
    line_df['Amp2'] = fit_info['%s_Amp2' % line]
    line_df['Cen2'] = fit_info['%s_Cen2' % line]
    line_df['Width2'] = fit_info['%s_Width2' % line]
    line_df['vel_model'] = fit_info['vel_model']

    sep = separate_components(line_df, restfreq)
    
    return sep


def create_vel_cube(mycube, restfreq, i, outfile):
    
    """"
    i = cube slice
    """
    
    mycube_dat = pyfits.getdata(mycube)

    # convert to velocity
    mycube_dat = SpectralCoord(mycube_dat[i], unit='AA')
    mycube_vel = (mycube_dat.to(u.km / u.s, doppler_rest = restfreq * u.AA, 
                                doppler_convention='optical'))
    # mycube_vel = np.where(np.array(mycube_vel) < -1000., np.nan, mycube_vel)
    
    # make it in the right format for fits
    get_hdr = pyfits.open(mycube)
    w = wcs.WCS(get_hdr[0].header,naxis=2).celestial
    new_header = w.to_header()
    #hdu = fits.PrimaryHDU(data=mycube_vel,header=new_header)
    fits.writeto(outfile, np.array(mycube_vel), new_header, overwrite=True)
   
    return


if __name__ == '__main__':
    
    # files and file names
    VelModelFile = 'NGC253.CO_1-0.diskfit.total_model_NKrieger.fits'
    VelModelReprojFile = 'ngc253_NKrieger_reproject.fits'
    FitInfoFile = 'ngc253_se_nosii_table.csv'
    MyCubeName  = 'ngc253_se_nosii'
    VelCubeName = 'ngc253_NKrieger'
    FitFiles = ['fit_1.fits','fit_2.fits','fit_3.fits','fit_4.fits']
    FitTotalFile = 'fit_total.fits'
    
    # get line information
    lines = ['Ha', 'NIIa', 'NIIb']
    restfreqs = [6564.61, 6549.86, 6585.27]
    
    # get cube info
    x = 437
    y = 436
    
    # user parameters
    plots_only = False
    
    if plots_only is False:
        
        # stitch em together
        #stitch(FitFiles,FitTotalFile)
        
        # open the full fit cube and then reproject the Krieger cube onto that footprint
        myfits = pyfits.open(FitTotalFile)
        velcube  = pyfits.open(VelModelFile)
        reproject_velcube(myfits,velcube,mycube_name=MyCubeName,
                          velcube_name=VelCubeName)
        
        # make the big table
        BigTable(pyfits.getdata(FitTotalFile), mycube_name=MyCubeName)
        
        # combine the fit info and the velocities from Krieger
        fit_info = pd.read_csv(FitInfoFile)
        vel_info = pyfits.getdata(VelModelReprojFile)
        fit_info['vel_model'] = vel_info.flatten()
        
        # generate maps of the outflow and disk for each line
        for i in range(len(lines)):
            
            print('Processing ' + lines[i] + '....')
            
            # get the line info into a DataFrame
            df = line_df(line=lines[i],restfreq = restfreqs[i], 
                                fit_info=fit_info)
            
            # generate maps for the outflow and the disk
            mapp(x=x, y=y, myfits=myfits, info=df, line=lines[i], 
                 feature='disk')
            
            mapp(x=x, y=y, myfits=myfits, info=df, line=lines[i], 
                 feature='outflow')
        
    # if plots_only is True     
    else:   
        files = ['nosii_outflow_flux_Ha.fits', 'nosii_disk_flux_Ha.fits',
          'nosii_outflow_vel_Ha.fits', 'nosii_disk_vel_Ha.fits',
          'nosii_outflow_fwhm_Ha.fits', 'nosii_disk_fwhm_Ha.fits',
          'nosii_outflow_flux_NIIa.fits', 'nosii_disk_flux_NIIa.fits',
          'nosii_outflow_vel_NIIa.fits', 'nosii_disk_vel_NIIa.fits',
          'nosii_outflow_fwhm_NIIa.fits', 'nosii_disk_fwhm_NIIa.fits',
          'nosii_outflow_flux_NIIb.fits', 'nosii_disk_flux_NIIb.fits',
          'nosii_outflow_vel_NIIb.fits', 'nosii_disk_vel_NIIb.fits',
          'nosii_outflow_fwhm_NIIb.fits', 'nosii_disk_fwhm_NIIb.fits']
        
        # files = ['subcube_outflow_flux_SIIa.fits','subcube_disk_flux_SIIa.fits',
        # 'subcube_outflow_vel_SIIa.fits','subcube_disk_vel_SIIa.fits',
        #  'subcube_outflow_fwhm_SIIa.fits','subcube_disk_fwhm_SIIa.fits',
        #  'subcube_outflow_flux_SIIb.fits','subcube_disk_flux_SIIb.fits',
        #  'subcube_outflow_vel_SIIb.fits','subcube_disk_vel_SIIb.fits',
        #  'subcube_outflow_fwhm_SIIb.fits','subcube_disk_fwhm_SIIb.fits']
        
        for file in files:
            # plot!
            gc = aplpy.FITSFigure(file)
            
            if 'vel' in file:
                if 'disk' in file:
                    gc.show_colorscale(vmin=120,vmax=300,cmap='RdBu_r')
                elif 'outflow' in file:
                    gc.show_colorscale(vmin=-100,vmax=150,cmap='RdBu_r')
                gc.add_colorbar()
                gc.colorbar.set_axis_label_text(axis_label_text='$v$ [km/s]')
            elif 'flux' in file:
                gc.show_colorscale(cmap='gist_rainbow')    
                gc.add_colorbar()
                gc.colorbar.set_axis_label_text(axis_label_text=
                                    'Flux [$10^{-20}$ erg/s/cm$^{2}$/$\AA$]')
            else:
                gc.show_colorscale(vmin=0,vmax=9,cmap='gist_rainbow')   
                gc.add_colorbar()
                gc.colorbar.set_axis_label_text(axis_label_text=
                                                'FWHM [$\AA$]')
            gc.set_title(file, fontsize=22)
            gc.colorbar.set_font(size=18)
            gc.colorbar.set_axis_label_font(size=22)
            gc.colorbar.set_axis_label_rotation(-90)
            gc.colorbar.set_axis_label_pad(30)
            gc.axis_labels.set_font(size=18)
            gc.tick_labels.set_font(size=18)
        
    
