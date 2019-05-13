import os
import wget

surveys = ["data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/",
           "data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_2/spectra/",
           "data.sdss3.org/sas/dr12/sdss/spectro/redux/26/spectra/",
           "data.sdss3.org/sas/dr12/sdss/spectro/redux/103/spectra/",
           "data.sdss3.org/sas/dr12/sdss/spectro/redux/104/spectra/"
          ]

def get_spec(row, save_dir):
	plate = str(row['plate']).zfill(4)
	mjd = str(row['mjd'])
	fiber_id = str(row['fiberid']).zfill(4)
	path = plate + '/spec-'+ plate + '-' + mjd + '-' + fiber_id + '.fits'
	down = False
	for surv in surveys:
		try:
			url = 'http://'+surv+path
			out_file = wget.download(url, out=save_dir)
			return None
		except:
			pass
	return row