# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:21:40 2020

@author: clermont
"""

import sys
sys.path.insert(0,'./src') 
import silknow_image_retrieval as sir

if __name__ == '__main__':
    configFile = sys.argv[1]
    sir.evaluate_model_configfile(configFile)
