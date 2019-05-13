CSS DR1/ SDSS DR14 CROSS-MATCH

OBJECTIVE
This folders makes a Bulk Search on SDSS DR14 available Optical Spectra (https://dr14.sdss.org/optical/spectrum/search), to check if there's cross-match with CSS DR1. The .fits (spectrum) files are downloaded for objects which are found to have a cross-match.

REQUIRED TOOLS
1. Selenium
2. Webdriver: http://chromedriver.chromium.org

INSTRUCTIONS
1. Run 'Build-batches.ipynb':
    1.1. Build batches of time series 'ra', 'dec' from CSS DR1 'photometry.csv' ('../../css/CSDR1/photometry.csv'), to query 'Bulk     Search' in Optical Spectra.
    1.2. Output: './Batches'. Batches are in directories to run multiple instances of the driver if needed.

2. Run 'Scraping.ipynb':
    2.1 Downloads any spectrum .fits found in a cross-match with a time series of any batches
    2.2. Ids are not saved when downloaded: local cross-match needs to be done after.

3. Run 'X-match.ipynb':
    3.1. Uses astropy to find the cross-match between the downloaded spectra (in '../spectra/') 
    with the time series in '../../css/CSDR1/photometry.txt'.
    3.2. Labels are gathered from CSS DR1 'catalog.txt' ('../../css/CSDR1/catalog.txt').
    3.2. Outputs: 
        3.2.1 DataFrame with: ['sloan_file', 'css_num_ID', 'label'] columns.
        3.2.2 A folder with each time series in cross-match ('../ts/').