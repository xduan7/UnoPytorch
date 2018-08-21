""" 
    File Name:          UnoPytorch/tee.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               8/17/18
    Python Version:     3.6.6
    File Description:   
        This file implements a helper class Tee, which redirects the stdout
        to a file while keeping things printed in console.
"""
import os
import sys


class Tee(object):

    def __init__(self, log_name, mode='a'):

        self._stdout = sys.stdout

        self._log_name = log_name
        self._mode = mode

        try:
            os.makedirs(os.path.dirname(log_name))
        except FileExistsError:
            pass

    def __del__(self):
        sys.stdout = self._stdout

    def write(self, data):

        # self._file = open(self._log_name, self._mode)
        # self._file.write(data)
        # self._file.flush()
        # self._file.close()

        with open(self._log_name, self._mode) as file:
            file.write(data)

        self._stdout.write(data)
        # self._stdout.flush()

    def flush(self):
        self._stdout.flush()

    def default_stdout(self):
        return self._stdout