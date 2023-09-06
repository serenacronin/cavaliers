

import sys
sys.path.append('../astro_tools')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import wcs
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import ImageGrid
from reproject import reproject_interp
from astropy import units as u
from spectral_cube import SpectralCube
import pyspeckit
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.wcs.utils import pixel_to_skycoord
import pvextractor

plt.rcParams['text.usetex'] = False
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["axes.labelweight"] = 'bold'
plt.rcParams["axes.titleweight"] = 'bold'
plt.rcParams["font.family"] = "courier new"
plt.rcParams["font.style"] = "normal"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["font.weight"] = 'bold'


def slice_extractor(filename, x_coord, y_coord, length, pa, width, velocity_convention):

    # read in the data cube
    cube = SpectralCube.read(filename).with_spectral_unit(u.km/u.s, velocity_convention=velocity_convention)
    cubew = cube[0,:,:].wcs

    # slice from the center coordinate
    coord = pixel_to_skycoord(x_coord, y_coord, wcs=cubew)
    path = pvextractor.PathFromCenter(center=coord,
                        length=length * u.arcsec,
                        angle=pa * u.deg,
                        width=width*u.arcsec)  # 10 pixels wide    
    myslice = pvextractor.extract_pv_slice(cube=cube, path=path)

    # grab each spectrum along the transverse axis (i.e., along the length of the slice)
    spectra_list = [myslice.data[:,i] for i in range(myslice.shape[1])]

    # grab the spatial axis of the PV slice and convert from degrees to arcsec
    spatial_axis_deg = wcs.WCS(myslice.header).array_index_to_world_values(np.zeros(myslice.shape[1]), np.arange(myslice.shape[1]))
    spatial_axis_arcsec = (spatial_axis_deg[0] * u.deg).to(u.arcsec)

    return myslice, spectra_list, spatial_axis_arcsec