# ICA-sEMG

sEMG signals contain cross talk, and a blind source separation algorithm is needed to identify the source signals.

We use the ICA algorithm[[1]](#1) to separate a set of sEMG source signals from a set of mixed sEMG signals, the implementation of the ICA algorithm is done using python-picard[[2-3]](#2-#3)

## How To Use
`ica_analysis` includes: 
- identification of the source signals 
- time plot of the source signals (with the option to scroll horizontally and/or include annotations) 
- heat maps of the source signals 
- spectral density plots of the source signals 

To use ica_analysis you need to convert your data to a numpy array, and import the script using: 
```python
from ica_analysis import ica
```

### Parameter Documentaion
`ica_analysis.ica(sigbufs, sampling_rate, heatmap_bool=False, image_path=None, hscroll_bool=False, annotations_bool=False, set_annotations=None,  set_times=None)
`
- **sigbufs** *(numpy array)*: signal data such that each row contains the data from a specific channel
- **sampling_rate** *(int)*: number of samples per second
- **heatmap_bool** *(bool, optional)*: boolean variable indicating whether you'd like to have the heatmap displayed (default is False)
- **image_path** *(string, optional)*: path of the image location for the heatmaps (default is None)
- **hscroll_bool** *(bool, optional)*: boolean variable indicating whether you'd like to have the horizontal scroll in the time plot displayed  (default is False)
- **annotations_bool** *(bool, optional)*: boolean variable indicating whether you'd like to have the annotations in the time plot displayed (default is False) 
- **set_annotations** *(numpy array, optional)*: annotation array (default is None)
- **set_times** *(numpy array, optional)*: annotation times array (default is None)

### sEMG Data Processing
Additionally the repo contains scripts to process sEMG data in two formats: 
- EDF: has the option to split the file to multiple, smaller files. The number of files is determined by "num_sets". Additionally user must fill out the path where the edf is stored ("edf_path")
- CSV: user must fill out the path of the CSV file 

## Required Python Libraries
- picard 
- scipy
- numpy
- matplotlib >= 3.4.2 
- math
- cv2  (for selecting electrode location)
- pandas


## References
<a id="1">[1]</a> 
A. Hyvärinen, E. Oja,
Independent component analysis: algorithms and applications,
Neural Networks,
Volume 13, Issues 4–5,
2000,
Pages 411-430,
ISSN 0893-6080,
https://doi.org/10.1016/S0893-6080(00)00026-5.



<a id="1">[2]</a> 
Pierre Ablin, Jean-Francois Cardoso, Alexandre Gramfort
Faster independent component analysis by preconditioning with Hessian approximations
IEEE Transactions on Signal Processing, 2018
https://arxiv.org/abs/1706.08171

<a id="1">[3]</a> 
Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort
Faster ICA under orthogonal constraint
ICASSP, 2018
https://arxiv.org/abs/1711.10873


