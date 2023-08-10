#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
Name: plot_fits.py

Created on Thu Jun  9 11:44:23 2022

Author: Serena A. Cronin

This script plots the resulting fits from routine.py.
========================================================================================
"""

import matplotlib.pyplot as plt
import warnings
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

def plotting(xpix, ypix, spec, redchisq, savepath, xmin, xmax, ymax, fluxnorm):

    """
    This function will plot the results of the fitting routine.
    It has the option to save every n fit to a directory and to include
    each component in the plot.

    Parameters
	-------------------

    xpix: int
        Pixel number along the x-axis.
    ypix: int
        Pixel number along the y-axis.
    xmin: int
        Lower end of the x-axis range.
    xmax: int
        Upper end of y-axis range.
    yresid: int
        Location on the y-axis to put the residuals. Recommended that this is < 0.
    fluxnorm: int
        Flux value for the y-axis to be normalized to.
    xlabel: str
        Label for the x-axis.
    ylabel: str
        Label for the y-axis.
    savepath: str
        Path to save the plots.
    plot_every: int
        Option to plot every n number of fits. Default is 1 (i.e., every fit).
    show_components: bool; default=True
        Option to plot the components of each fit. Default is True.
    
    """
    
    # set up the plotter
    spec.plotter(xmin = xmin, xmax = xmax, ymin = -0.4*ymax) 
    plt.rcParams["figure.figsize"] = (10,5)   
    spec.measure(fluxnorm = fluxnorm)
    
    # set axes labels and refresh the plotter
    spec.plotter.axis.set_xlabel(r'Wavelength $(\AA)$')
    spec.plotter.axis.set_ylabel(r'S$_{\lambda}$ $(10^{-20} \mathrm{erg/s/cm^2/\AA})$')
    
    # plot the fit, including the individual components
    spec.specfit.plot_fit(annotate=False, 
                          show_components=False,
                          composite_fit_color='tab:pink',
                          lw=1.5)
    
    spec.specfit.plot_components(component_fit_color='tab:cyan',
                                lw=1.5)
    
    # plot the residuals
    spec.specfit.plotresiduals(axis=spec.plotter.axis,
                                clear=False,
                                yoffset=-0.2*ymax, 
                                color='tab:purple',
                                linewidth=1.5)
    
    # get the fit information to the side of the plot
    spec.specfit.annotate(loc='upper right', labelspacing=0.15, markerscale=0.01, 
                          borderpad=0.1, handlelength=0.1, handletextpad=0.1, 
                          fontsize=6, bbox_to_anchor=(1.3,1.1))
    
    # adjust the plot so that the annotation can be seen
    plt.subplots_adjust(right=0.75)

    # make and plot a custom legend
    custom_lines = [Line2D([0], [0], color='tab:pink', lw=2),
                    Line2D([0], [0], color='tab:cyan', lw=2),
                    Line2D([0], [0], color='tab:purple', lw=2),
                    Line2D([0], [0], color='white', lw=2)]
    
    plt.legend(custom_lines,['Composite', 'Components', 'Residuals',
                            'RedChiSq: %s' % round(redchisq,2)], fontsize=7.5, 
                            loc='upper left')
        
    # make a title
    plt.title('Pixel: %s,%s' % (164+xpix,359+ypix))
    
    # plt.xlabel(r'Wavelength $(\AA)$')
    
    # save the figure and suppress it from being printed to the terminal
    plt.savefig('%s/pixel_%s_%s.png' % (savepath, 164+xpix,359+ypix), 
                dpi=200)
    plt.close()
    return