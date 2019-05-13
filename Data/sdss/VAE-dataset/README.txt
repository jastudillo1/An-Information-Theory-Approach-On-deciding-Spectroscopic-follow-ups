The collected spectra here is for training spectra feature extraction VAE model.

For collecting the spectra:
1. An SQL query is done in CasJobs (https://skyserver.sdss.org/casjobs/)
	1.1. 'query.sql' has the query done to CasJobs
	1.2. Source: DR12
	1.3. ~21,000 spectra, balanced OBAFGKM
	1.4. Queried columns: 'survey', 'subclass', 'plate', 'mjd', 'fiberid'
	1.5. Output as CSV: 'keys.csv'

2. With 'keys.csv' spectra is downloaded as follows:
	2.1. Running 'Download.ipynb'
	2.2. Spectra is saved in './spectra/' in fits format
	2.1 Download links:
	'data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_0/spectra/',
	'data.sdss3.org/sas/dr12/boss/spectro/redux/v5_7_2/spectra/',
	'data.sdss3.org/sas/dr12/sdss/spectro/redux/26/spectra/',
	'data.sdss3.org/sas/dr12/sdss/spectro/redux/103/spectra/',
	'data.sdss3.org/sas/dr12/sdss/spectro/redux/104/spectra/'
          ]
	2.1. Links were retrieved from 'Bulk Download Data'>'Optical Spectra Per-Object 	Files' from 'https://www.sdss.org/dr12/data_access/bulk/'
	2.2. For DR14 links: 'https://www.sdss.org/dr14/data_access/bulk/'
	2.3. Most DR12 data is contained in DR14 as indicated in 
	'https://www.sdss.org/dr14/ > 'Past Data Releases'