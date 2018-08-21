""" 
    File Name:          UnoPytorch/file_downloading.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/13/18
    Python Version:     3.6.6
    File Description:   

"""

import os
import urllib
import logging


logger = logging.getLogger(__name__)


def download_files(
        filenames: str or iter,
        ftp_root: str,
        target_folder: str, ):

    if type(filenames) is str:
        filenames = [filenames, ]

    # Download each file in the list
    for filename in filenames:
        file_path = os.path.join(target_folder, filename)

        if not os.path.exists(file_path):
            logger.debug('File does not exit. Downloading %s ...' % filename)

            url = ftp_root + filename
            try:
                url_data = urllib.request.urlopen(url)
                with open(file_path, 'wb') as f:
                    f.write(url_data.read())
            except IOError:
                logger.error('Failed to open and download url %s.' % url,
                             exc_info=True)
                raise
