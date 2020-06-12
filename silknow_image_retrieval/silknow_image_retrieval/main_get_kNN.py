# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:20:00 2020

@author: clermont
"""

import sys
sys.path.insert(0,'./src') 
import silk_retrieval_functions as srf

if __name__ == '__main__':
    configFile = sys.argv[1]
    srf.get_kNN_configfile(configFile)