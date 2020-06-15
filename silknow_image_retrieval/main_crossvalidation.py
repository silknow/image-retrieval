# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:48:39 2020

@author: clermont
"""

import sys
sys.path.insert(0,'./src') 
import silknow_image_retrieval as sir

if __name__ == '__main__':
    configFile = sys.argv[1]
    sir.crossvalidation_configfile(configFile)