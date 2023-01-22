## `CAVALIERS`: **C**ube **A**lgorithm for **V**ery **A**ccurate **L**ine **I**nvestigations of **E**mission in **R**adiant **S**uperwinds.

<img width="560" alt="logoCAVALIERS" src="https://user-images.githubusercontent.com/53054401/212378737-d3341850-3064-4061-b8e2-7f8ea914aefc.png">

A wrapper for the `pyspeckit` package (https://pyspeckit.readthedocs.io/en/latest/) that fits Gaussian models to emission lines of outflows in nearby galaxies. Note: within this wrapper are handy tools that deal with fitting data cubes that might be useful for other science cases.

This is being tested and implemented on a project that analyzes MUSE data cubes of the central region of NGC 253. We are fitting three emission lines, each of which have two separate Gaussian peaks that correspond to contributions from both the wind and the disk of the galaxy. The algorithm involves testing out a different number of Gaussian components and comparing them using statistical analyses in order to find the best fit for each pixel.

## Required packages:

  `spectral-cube`
  `pyspeckit`
  `astropy`
  `numpy`
  `scipy`
  `matplotlib`
  `tqdm`
  `os`
  `sys`
  
