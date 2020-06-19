SILKNOW Image Retreival
===============================

version number: 0.0.1
author: LUH

Overview
--------

This software provides Python functions for the retrieval of images and for training and evaluation of CNN-based retrieval models. It consists of six main parts:
	1. creation of a dataset
	2. training of a new retrieval model
	3. creation of a descriptor index forming the search space for image retrieval
	4. retrieval of images using an existing model and according index
	5. evaluation of an existing retrieval model and the combined training
	6. evaluation in a ﬁve-fold cross validation.
All functions take either conﬁguration ﬁles or explicit parameter settings as an input and generally write their results in speciﬁed paths. The format required for the conﬁguration ﬁles is described in the SILKNOW Deliverable D4.5.


Installation / Usage
--------------------

To install use:

	$ git clone https://github.com/silknow/silknow_image_retrieval.git
    $ pip install --upgrade .


Or:

    $ git clone https://github.com/silknow/silknow_image_retrieval.git
    $ python setup.py install
  
The requirements for the silknow_image_retrieval toolbox are Python 3.6.4 and the following Python packages that are automatically installed within the package installation:
	• numpy
	• urllib3
	• pandas (0.25.1)
	• tqdm
	• opencv-python
	• tensorﬂow (1.13.1)
	• tensorﬂow-hub (0.2.0)
	• matplotlib
	• sklearn
	• scipy  

A pre-trained model as well as the corresponding descriptor index that were both created using this software can be download from http://doi.org/10.5281/zenodo.3901091.