# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:48:39 2020

@author: clermont
"""

import sys
sys.path.insert(0,'./src') 
import silk_retrieval_functions as srf

if __name__ == '__main__':
    configFile = sys.argv[1]
    srf.crossvalidation_configfile(configFile)