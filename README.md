## `CAVALIERS`: **C**ube **A**lgorithm for **V**ery **A**ccurate **L**ine **I**nvestigation of **E**mission in **R**adiant **S**uperwinds.

Tools related to handling spectral cubes. Wrapper for the pyspeckit package (https://pyspeckit.readthedocs.io/en/latest/).

This is being tested and implemented on a project that analyzes MUSE data cubes of the central region of NGC 253. We are fitting three emission lines, each of which have two separate Gaussian peaks that correspond to contributions from both the wind and the disk of the galaxy. Therefore, this has successfully fit 6 Gaussians simultaneously at fixed separations.

## Required packages:

  `spectral-cube`
  `pyspeckit`
  `astropy`
  `numpy`
  `scipy`
  `matplotlib`
  `tqdm`
  
