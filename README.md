SILKNOW Image Retrieval
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
    
All functions take explicit parameter settings as an input and generally write their results in speciﬁed paths. A documentation of the functions' parameters can be found in [documentation](https://github.com/silknow/image-retrieval/tree/master/silknow_image_retrieval/documentation) and further details are described in the SILKNOW Deliverable D4.6.


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

A pre-trained model that was created using this software can be download from https://doi.org/10.5281/zenodo.4745316. The training of that model is based on the `combined_similarity_loss` with equal weights for all of the `loss_weight_*`.

 User Guidelines
-----------------

The user can either download the [provided image retrieval model](https://doi.org/10.5281/zenodo.4745316) and build an own descriptor index using the function `silknow_image_retrieval.build_kDtree_parameter` and the images of an own database. Afterwards, image retrieval for new images can be realized by means of the function `silknow_image_retrieval.get_kNN_parameter`.

Alternatively, the user can train an own image retrieval model using the provided software for a subsequent descriptor building and nearest neighbour search. Therefore, example calls of all functions are provided in [main.py](https://github.com/silknow/image-retrieval/blob/master/silknow_image_retrieval/main.py) using the [provided data files](https://github.com/silknow/image-retrieval/tree/master/silknow_image_retrieval/samples). These function calls will perform all steps listed in the overview above using the [provided knowledge graph export](https://github.com/silknow/image-retrieval/blob/master/silknow_image_retrieval/samples/total_post.csv).