README

1. Build-db: 
	- This script prepares data for training a VAE which learns feature extraction from spectra.
	- ~21,000 spectra from SDSS dr12 are used as explained in '../../sdss/VAE-dataset/README.txt'
	- All spectrum are preprocessed to share the same wavelengths:
		- All wavelengths start points are sorted. 
		- A common starting wavelength is set to the value of the wavelength in percentil 99%.
		- All spectra which have a higher starting point are dropped.
		- Same is done for the ending wavelength.
		- The resulting starting and ending wavelength are 3,830 and 9,174 Angstrom, respectively.
		  (these may be checked in './data/params.npy).
		- Coverage may be different (starting and ending wavelength) but resolution is indicated in 
		  http://www.sdss3.org/dr9/spectro/spectro_basics.php.
		- No need to bin since all spectra in SDSS have measures in exactly same wavelengths. 
	- Each spectrum Flux is normalized to 0-1 range (MinMaxScaler). 
	  The normalization is not done per wavelength measure or bin but for each spectrum.