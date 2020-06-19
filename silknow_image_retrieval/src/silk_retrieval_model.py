# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:43:10 2020

@author: clermont, dorozynski
"""
import numpy as np
import os
import tensorflow as tf
import sys
import tensorflow_hub as hub
import random
import math
import pandas as pd
import urllib
import urllib.request
import matplotlib.pyplot as plt
import cv2

from shutil import copy

from sklearn.neighbors import KDTree
from operator import itemgetter
from scipy import stats
from tqdm import tqdm


sys.path.insert(0, '../../')
sys.path.insert(0, '../')
sys.path.insert(0, './src/')
from . import SILKNOW_WP4_library as sn_func
CHECKPOINT_NAME = 'model.ckpt'


def create_dataset(csvfile,
                   imgsavepath = "../data/",
                   minnumsamples = 150,
                   retaincollections = ['garin', 'imatex', 'joconde', 'mad', 'mfa', 'risd']):
    r"""Creates the dataset for the CNN.

    :Arguments\::
        :csvfile (*string*)\::
            Filename (including the path) of the .csv file that represents all data used
            for training and testing the classifier.
        :imgsavepath (*string*)\::
            Path to where the images will be downloaded. This path has
            to be relative to the main software folder.
        :minnumsaples (*int*)\::
            Minimum number of samples for each class. Classes with fewer
            occurences will be ignored and set to unknown.
        :retaincollections (*list*)\::
            List of strings that defines the museums/collections that
            are be used. Data from museums/collections
            not stated in this list will be omitted.


    :Returns\::
        No returns. This function produces all files needed for running the software.

    """
    # TODO: D4.6: automatically create table about data statistics

    MasterfilePath = r"./samples/"

    # read file
    df = pd.read_csv(csvfile)

    # obj auf URI kürzen
    df.obj = df.obj.apply(lambda x: x.split("/")[-1])

    # Museum auf Name kürzen
    df.museum = df.museum.apply(
        lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(",")[0].split("/")[-1])

    # URLs und Bildnamen zu Listen konvertieren
    df.deeplink = df.deeplink.apply(lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    df.img = df.img.apply(lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    df.img = df.img.apply(lambda x: list(map(lambda y: y.split("/")[-1], x)))

    # Arrange entries into lists
    df.place_country_code = df.place_country_code.apply(
        lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    df.place_country_code = df.place_country_code.apply(lambda x: x[0] if len(x) == 1 else np.nan).apply(
        lambda x: np.nan if x == 'nan' else x)

    df.time_label = df.time_label.apply(lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    df.time_label = df.time_label.apply(lambda x: x[0] if len(x) == 1 else np.nan).apply(
        lambda x: np.nan if x == 'nan' else x)

    df.technique_group = df.technique_group.apply(
        lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    df.technique_group = df.technique_group.apply(lambda x: x[0] if len(x) == 1 else 'nan')
    df.technique_group = df.technique_group.apply(lambda x: x.split("/")[-1] if not x == 'nan' else np.nan)

    df.material_group = df.material_group.apply(
        lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    df.material_group = df.material_group.apply(lambda x: x[0] if len(x) == 1 else 'nan')
    df.material_group = df.material_group.apply(lambda x: x.split("/")[-1] if not x == 'nan' else np.nan)

    df.depict_group = df.depict_group.apply(
        lambda x: x.strip("[").strip("]").replace("', '", ",").strip("''").split(","))
    df.depict_group = df.depict_group.apply(lambda x: x[0] if len(x) == 1 else 'nan')
    df.depict_group = df.depict_group.apply(lambda x: x.split("/")[-1] if not x == 'nan' else np.nan)

    # Drop unused columns
    df = df.drop(labels=["place_uri", "time_uri"], axis='columns')

    # Vorverarbeitung für Records, ein Record pro Bild
    totallist = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # row = row_[1]
        varlist = [[row.museum + "__" + row.obj + "__" + URL.split('/')[-1],
                    URL,
                    row.place_country_code,
                    row.time_label,
                    row.material_group,
                    row.technique_group,
                    row.depict_group,
                    row.museum,
                    row.obj] for URL in row.deeplink]
        totallist += varlist
    tl = np.asarray(totallist).transpose()

    # Erstelle Datensatz
    data = pd.DataFrame({'ID': tl[0],
                         'URL': tl[1],
                         'place': tl[2],
                         'timespan': tl[3],
                         'material': tl[4],
                         'technique': tl[5],
                         'depiction': tl[6],
                         'museum': tl[7],
                         'obj': tl[8]}).replace([""], np.nan).replace("nan", np.nan)
    # 'obj': tl[7]}).set_index("ID").replace([""], np.nan).replace("nan", np.nan)

    # Throw out (i.e. set to nan) all values occuring fewer than 150 times
    for c in ["place", "timespan", "material", "technique", "depiction"]:
        names = data[c].value_counts().index.tolist()
        count = data[c].value_counts().tolist()
        for na, co in zip(names, count):
            if co < minnumsamples:
                data[c] = data[c].replace(na, np.nan)

    # count NaNs
    variable_list = ["timespan", "place", "material", "technique", "depiction"]
    data['nancount'] = data[variable_list].isnull().sum(axis=1)

    # omit all records from museums with too many non-fabrics...
    # oklist = ['garin', 'imatex', 'joconde', 'mad', 'mfa', 'risd']
    oklist = retaincollections
    data = data[data.museum.isin(oklist)]

    # ... as well as records with no information
    data = data[data.nancount < len(variable_list)]

    # Download der Daten
    # if download:
    if True:
        # savepath = "./img_unscaled/"
        # checkpath = "./img_unscaled/"
        savepath = checkpath = imgsavepath + r"img_unscaled/"
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        deadlinks = 0
        deadlist = []
        for index, row in tqdm(data.iterrows(), total=len(data.index)):

            # Skip record if image already exists
            if os.path.isfile(checkpath + row.ID): continue

            # Try to download from URL until one URL works
            url = row.URL
            try:
                urllib.request.urlretrieve(url, savepath + row.ID)
            except:
                deadlinks += 1
                deadlist += [url]
        print("In total,", deadlinks, "records have no functioning image link!")
    #     print(deadlist)

    # Rescaling
    # if rescale:
    if True:
        # Rescaling of downloaded images
        imgpath_load = imgsavepath + r"/img_unscaled/"
        imgpath_save = imgsavepath + r"img/"
        if not os.path.exists(imgpath_save):
            os.makedirs(imgpath_save)

        imglist = os.listdir(imgpath_load)

        # for img_file in tqdm(imglist):
        deadlist_load = []
        deadlist_scale = []
        for img_file in tqdm(imglist, total=len(imglist)):

            # Skip images that are already scaled
            if os.path.exists(imgpath_save + img_file): continue

            # Try to open images, skip else
            try:
                img = plt.imread(imgpath_load + img_file)
            except:
                deadlist_load += [img_file]
                continue

            try:
                # get dimensions of image
                if len(img.shape) == 2:
                    width, heigth = img.shape
                elif len(img.shape) == 3:
                    width, heigth, _ = img.shape

                smaller_side = np.minimum(heigth, width)
                scale_factor = 448. / smaller_side

                # If Downscaling, apply gaussian blur
                if scale_factor < 1.:
                    sigma = 1. / scale_factor
                    kernelsize = int(sigma * 6) + (1 - (int(sigma * 6) % 2))
                    img = cv2.GaussianBlur(img, (kernelsize, kernelsize), sigma)

                img_new = cv2.resize(img, (int(heigth * scale_factor), int(width * scale_factor)),
                                     interpolation=cv2.INTER_CUBIC)
                plt.imsave(imgpath_save + img_file, img_new)
            except:
                # print("Fehler beim Skalieren/Speichern:"+img_file)
                deadlist_scale += [img_file]
        print("Couldn't load:", deadlist_load)
        print("Couldn't scale:", deadlist_scale)

    # only retain records with existing images
    mypath = imgsavepath + r"img/"
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    onlyfiles = list(dict.fromkeys(onlyfiles))
    data = data.set_index("ID").loc[[f for f in onlyfiles]]

    # Check again for minimum of 150 occurrences
    for c in variable_list:
        names = data[c].value_counts().index.tolist()
        count = data[c].value_counts().tolist()
        for na, co in zip(names, count):
            if co < 150:
                data[c] = data[c].replace(na, np.nan)

    # count NaNs
    data['nancount'] = data[variable_list].isnull().sum(axis=1)

    # Omit again records with only missing annotations
    data = data[data.nancount < len(variable_list)]

    # re-name labels to not include any spaces
    for c in variable_list:
        label_list = data[c].unique()
        for l in label_list:
            if " " in str(l):
                changed_label = l.replace(" ", "_")
                data = data.replace(l, changed_label)

    # print statistics
    # [print(data[var].value_counts(dropna=False)) for var in data.columns]

    # create collection files
    image_data = data.sample(frac=1)
    dataChunkList = np.array_split(image_data, 5)
    variable_list = ["place", "timespan", "material", "technique", "depiction", "museum"]

    # get complete samples
    complete_mask = data.loc[data["nancount"] == 0]
    complete_mask.to_csv("complete.csv")

    for i, chunk in enumerate(dataChunkList):
        collection = open(MasterfilePath + "collection_" + str(i + 1) + ".txt", "w+")
        #        string = ["#"+name+"\t" for name in list(image_data)[1:]]
        string = ["#" + name + "\t" for name in variable_list]
        collection.writelines(['#image_file\t'] + string + ["\n"])

        for index, row in chunk.iterrows():
            imagefile = str(row.name) + ".jpg\t" if not ".jpg" in str(row.name) else str(row.name) + "\t"

            # Skip improperly formatted filenames
            if "/" in imagefile: continue

            #            string = [(str(row[label])+"\t").replace('nan','NaN') for label in list(image_data)[1:]]
            string = [(str(row[label]) + "\t") for label in variable_list]

            collection.writelines([r"../" + imgsavepath + r"img/" + imagefile] + string + ["\n"])

        collection.close()

    # # Write collection files to masterfile and save it in the same path
    # master = open(MasterfilePath + "Masterfile.txt", "w+")
    # for i in range(len(dataChunkList)):
    #     master.writelines(["collection_"] + [str(i + 1)] + [".txt\n"])
    # master.close()

    # Print label statistics
    classStructures = {}
    for v in variable_list:
        print("Classes for variable", v)
        print(image_data[v].value_counts(dropna=False))
        labels = image_data[v].unique()
        classStructures[v] = labels[~pd.isnull(labels)]
        print("\n")

    # print label statistics per collection
    for c in retaincollections:
        vardf = image_data[image_data.museum == c]
        for v in variable_list:
            print("Classes for variable", v, " in museum", c, ":")
            print(vardf[v].value_counts(dropna=False))
            print("\n")

        print("\n")
    #
    # # save pandas dataframe to csv
    # image_data.to_csv("image_data.csv")

    ######### Different to D4.6 ################

    # # Write statistics to file
    # for c in retaincollections:
    #     vardf = image_data[image_data.museum == c]
    #     for v in variable_list:
    #         print("Classes for variable", v, " in museum", c, ":")
    #         print(vardf[v].value_counts(dropna=False))
    #         print("\n")
    # data = df
    # with pd.ExcelWriter('classes_' + '.xlsx') as writer:
    #     for var in data.columns:
    #         data[var].value_counts(dropna=False).to_excel(writer, sheet_name=var)

    # Write collection files to masterfile and save it in the same path
    master = open(MasterfilePath + "masterfile.txt", "w+")
    for i in range(len(dataChunkList)):
        master.writelines(["collection_"] + [str(i + 1)] + [".txt\n"])
    master.close()

    # create further master files for tree building and get kNN
    master = open(MasterfilePath + "master_file_tree.txt", "w+")
    for i in range(len(dataChunkList) - 1):
        master.writelines(["collection_"] + [str(i + 1)] + [".txt\n"])
    master.close()

    master = open(MasterfilePath + "master_file_prediction.txt", "w+")
    master.writelines(["collection_"] + [str(len(dataChunkList))] + [".txt\n"])
    master.close()

    # Create default configuration files
    create_config_file_train_model(MasterfilePath, variable_list)
    create_config_file_build_tree(MasterfilePath, variable_list)
    create_config_file_get_kNN(MasterfilePath)
    create_config_file_evaluate_model()
    create_config_file_crossvalidation(MasterfilePath, variable_list)


def create_config_file_train_model(masterfile_path, variable_list):
    """

    """
    config = open("Configuration_train_model.txt", "w+")

    config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
    config.writelines(["master_file_name; master_file_tree.txt\n"])
    config.writelines(["master_dir; " + masterfile_path + "\n"])
    config.writelines(["logpath; " + r"./output_files/Default/" + "\n"])

    config.writelines(["\n****************CNN ARCHITECTURE SPECIFICATIONS**************** \n"])
    config.writelines(["add_fc; [1024, 128] \n"])
    config.writelines(["hub_num_retrain; 0 \n"])

    config.writelines(["\n****************TRAINING SPECIFICATIONS**************** \n"])
    config.writelines(["train_batch_size; 150\n"])
    config.writelines(["how_many_training_steps; 200\n"])
    config.writelines(["learning_rate; 1e-4\n"])
    config.writelines(["val_percentage; 25\n"])
    config.writelines(["how_often_validation; 10\n"])
    config.writelines(["loss_ind; soft_contrastive_incomp_loss\n"])

    config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
    config.writelines(["relevant_variables; "])
    for variable in variable_list[0:-2]:
        config.writelines(["#%s, " % str(variable)])
    config.writelines(["#%s\n" % str(variable_list[-2])])

    config.writelines(["\n****************DATA AUGMENTATION SPECIFICATIONS**************** \n"])
    config.writelines(["random_crop; [0.7, 1]\n"])
    config.writelines(["random_rotation90; True\n"])
    config.writelines(["gaussian_noise; 0.01\n"])
    config.writelines(["flip_left_right; True\n"])
    config.writelines(["flip_up_down; True\n"])
    config.close()


def create_config_file_build_tree(masterfile_path, variable_list):
    """

    """
    config = open("Configuration_build_kDTree.txt", "w+")

    config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
    config.writelines(["model_path; " + r"./output_files/Default/" + "\n"])
    config.writelines(["master_file_tree; master_file_tree.txt\n"])
    config.writelines(["master_dir_tree; " + masterfile_path + "\n"])
    config.writelines(["savepath; " + r"./output_files/Default/" + "\n"])

    config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
    config.writelines(["relevant_variables; "])
    for variable in variable_list[0:-2]:
        config.writelines(["#%s, " % str(variable)])
    config.writelines(["#%s" % str(variable_list[-2])])
    config.close()


def create_config_file_get_kNN(masterfile_path):
    """

    """
    config = open("Configuration_get_kNN.txt", "w+")

    config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
    config.writelines(["treepath; " + r"./output_files/Default/" + "\n"])
    config.writelines(["master_file_prediction; master_file_prediction.txt\n"])
    config.writelines(["master_dir_prediction; " + masterfile_path + "\n"])
    config.writelines(["model_path; " + r"./output_files/Default/" + "\n"])
    config.writelines(["savepath; " + r"./output_files/Default/" + "\n"])

    config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
    config.writelines(["num_neighbors; 6\n"])
    config.writelines(["bool_labeled_input; True\n"])
    config.close()


def create_config_file_evaluate_model():
    """

    """
    config = open("Configuration_evaluate_model.txt", "w+")

    config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
    config.writelines(["pred_gt_path; " + r"./output_files/Default/" + "\n"])
    config.writelines(["result_path; " + r"./output_files/Default/" + "\n"])
    config.close()


def create_config_file_crossvalidation(masterfile_path, variable_list):
    """

    """
    config = open("Configuration_crossvalidation.txt", "w+")

    config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
    config.writelines(["master_file_name; masterfile.txt\n"])
    config.writelines(["master_dir; " + masterfile_path + "\n"])
    config.writelines(["logpath; " + r"./output_files/Default/" + "\n"])

    config.writelines(["\n****************CNN ARCHITECTURE SPECIFICATIONS**************** \n"])
    config.writelines(["add_fc; [1024, 128] \n"])
    config.writelines(["hub_num_retrain; 0 \n"])

    config.writelines(["\n****************TRAINING SPECIFICATIONS**************** \n"])
    config.writelines(["train_batch_size; 150\n"])
    config.writelines(["how_many_training_steps; 200\n"])
    config.writelines(["learning_rate; 1e-4\n"])
    config.writelines(["val_percentage; 25\n"])
    config.writelines(["how_often_validation; 10\n"])
    config.writelines(["loss_ind; soft_contrastive_incomp_loss\n"])

    config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
    config.writelines(["num_neighbors; 6\n"])
    config.writelines(["relevant_variables; "])
    for variable in variable_list[0:-2]:
        config.writelines(["#%s, " % str(variable)])
    config.writelines(["#%s\n" % str(variable_list[-2])])

    config.writelines(["\n****************DATA AUGMENTATION SPECIFICATIONS**************** \n"])
    config.writelines(["random_crop; [0.7, 1]\n"])
    config.writelines(["random_rotation90; True\n"])
    config.writelines(["gaussian_noise; 0.01\n"])
    config.writelines(["flip_left_right; True\n"])
    config.writelines(["flip_up_down; True\n"])
    config.close()


def crossvalidation(masterfile_name_crossval, masterfile_dir_crossval, logpath,
                    train_batch_size, how_many_training_steps, learning_rate,
                    tfhub_module, add_fc, hub_num_retrain,
                    aug_dict, optimizer_ind, loss_ind,
                    relevant_variables, similarity_thresh, label_weights, 
                    how_often_validation, val_percentage, num_neighbors):
    r""" Performs crossvalidation.
    
    :Arguments\::
        :master_file (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
        :master_dir (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :logpath (*string*):
            The path where all summarized training information and the trained
            network will be stored.
        :train_batch_size (*int*)\::
            This variable defines how many images shall be used for
            the classifier's training for one training step.
            Default: XXX.
        :how_many_training_steps (*int*)\::
            Number of training iterations.
        :learning_rate (*float*)\::
            Specifies the learning rate of the Optimizer.
            Default: XXX.
        :tfhub_module (*string*)\::
            This variable contains the Module URL to the
            desired networks feature vector. For ResNet-152 V2 is has to be
            'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1'.
            Other posibilities for feature vectors can be found at
            'https://tfhub.dev/s?module-type=image-feature-vector'.
            Default: XXX.
        :add_fc (*array of int*)\::
            The number of fully connected layers that shall be trained on top
            of the tensorflow hub Module is equal to the number of entries in
            the array. Each entry is an int specifying the number of nodes in
            the individual fully connected layers. If [1000, 100] is given, two
            fc layers will be added, where the first has 1000 nodes and the
            second has 100 nodes. If no layers should be added, an empty array
            '[]' has to be given.
            Default: XXX.
        :hub_num_retrain (*int*)\::
            The number of layers from the
            tensorflow hub Module that shall be retrained. 
            Default: XXX.
        :aug_dict (*dict*)\::
            A dictionary specifying which types of data augmentations shall 
            be applied during training. A list of available augmentations can be
            found in the documentation of the SILKNOW WP4 Library.
            Default: XXX
        :optimizer_ind (*string*)\::
            The optimizer that shall be used during the training procedure of
            the siamese network. Possible options:
                'Adagrad' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
                'Adam' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
                'GradientDescent' (cf. https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)
            Default: XXX
        :loss_ind (*string*)\::
            The loss function that shall be utilized to estimate the network
            performance during training. Possible options:
                'contrastive'       (cf. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964)
                'soft_contrastive' (own development)
                'contrastive_thres' (own development)
                'triplet_loss'      (cf. https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
                'soft_triplet'  (own development; ~triplet_multi)
                'triplet_thresh'    (own development) 
            Default: XXX
        :relevant_variables (*list*)\::
            A list containing the collection.txt's header terms of the labels
            to be considered. The terms must not contain blank spaces; use '_'
            instead of blank spaces!
            Example (string in control file): #timespan, #place  
            Example (according list in code): [timespan, place]
            Default: XXX
        :similarity_thresh (*float*)\::
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
            Default: XXX
        :label_weights (*list*)\::
            Weights that express the importance of the individual labels for
            the similarity estimation. Have to be positive numbers. Will be
            normalized so that the sum is 1.
            The list has to be as long as "relevant_variables".
            Default: XXX
        :how_often_validation (*int*)\::
            Number of training iterations between validations.
            Default: XXX
        :val_percentage (*int*)\::
            Percentage of training data that will be used for validation.
            Default: XXX
        :num_neighbors (*int*)\::
            Number of closest neighbours that are retrieved from a kD-Tree
            during evaluation.
            Default: XXX

    """
    
    # make sure path exists
    if not os.path.exists(os.path.join(logpath,r"")): os.makedirs(os.path.join(logpath,r""))
    
    # load masterfile 
    coll_list = sn_func.master_file_to_collections_list(masterfile_dir_crossval, masterfile_name_crossval)
    
    # averaging results from all cviters
    predictions = []
    groundtruth = []
    all_pred_top_k = []
    all_labels_target = []
    
    # FIVE cross validation iterations
    for cviter in range(5):
    
        # create intermediate masterfiles for sub-modules
        train_master = open(os.path.abspath(masterfile_dir_crossval+'/'+'train_master.txt'), 'w')
        test_master  = open(os.path.abspath(masterfile_dir_crossval+'/'+'test_master.txt'), 'w')
        train_coll = np.roll(coll_list, cviter)[:-1]
        test_coll  = np.roll(coll_list, cviter)[-1]
        for c in train_coll:
            train_master.write("%s\n" % (c))
        test_master.write(test_coll)
        train_master.close()
        test_master.close()
        
        # set sub-logpath
        logpath_cv = logpath+r"/cv"+str(cviter)+"/"
        if not os.path.exists(logpath_cv): 
            os.makedirs(logpath_cv)
        
        # perform training
        train_model(master_file = 'train_master.txt',
                    master_dir                 = masterfile_dir_crossval,
                    logpath                    = logpath_cv,
                    train_batch_size           = train_batch_size,
                    how_many_training_steps    = how_many_training_steps,
                    learning_rate              = learning_rate,
                    tfhub_module               = tfhub_module,
                    add_fc          = add_fc,
                    hub_num_retrain = hub_num_retrain,
                    aug_dict        = aug_dict,
                    optimizer_ind   = optimizer_ind,
                    loss_ind        = loss_ind,
                    relevant_variables = relevant_variables,
                    similarity_thresh    = similarity_thresh,
                    label_weights        = label_weights,
                    how_often_validation = how_often_validation,
                    val_percentage       = val_percentage)

        # build tree
        build_kDTree(model_path = logpath_cv,
                     master_file_tree = 'train_master.txt',
                     master_dir_tree  = masterfile_dir_crossval,
                     relevant_variables  = relevant_variables,
                     savepath = logpath_cv)

        # predictions
        (pred_top_k,
         labels_target,
         label2class_list) = get_kNN(treepath = logpath_cv,
                                     master_file_prediction = 'test_master.txt',
                                     master_dir_prediction = masterfile_dir_crossval,
                                     model_path = logpath_cv,
                                     bool_labeled_input = True,
                                     num_neighbors = num_neighbors,
                                     savepath = logpath_cv)
        
        # evaluations
        gtvar, prvar = evaluate_model(pred_gt_path = logpath_cv, 
                                      result_path  = logpath_cv)

        # concatenate predictions and groundtruths
        if len(predictions) == 0:
            predictions = prvar
            groundtruth = gtvar
            all_pred_top_k = pred_top_k
            all_labels_target = labels_target
        else:
            predictions = np.concatenate((predictions, prvar))
            groundtruth = np.concatenate((groundtruth, gtvar))
            all_pred_top_k = np.concatenate((all_pred_top_k, pred_top_k))
            all_labels_target = np.concatenate((all_labels_target, labels_target))
        
        # delete intermediate data
        os.remove(masterfile_dir_crossval+'/'+'train_master.txt')
        os.remove(masterfile_dir_crossval+'/'+'test_master.txt')

    # TODO: raus für D4.5
    # estimate top k statistics among all cross validation iterations
    get_top_k_statistics(all_pred_top_k, all_labels_target, num_neighbors,
                         logpath, label2class_list)

    # estimate quality measures with all predictions and groundtruths
    vardict = np.load(logpath_cv+r"/pred_gt.npz", allow_pickle=True)["arr_0"].item()
    label2class_list = np.asarray(vardict["label2class_list"])
    for task_ind, classlist in enumerate(label2class_list):
        taskname = classlist[0]
        list_class_names = classlist[1:]
        
        # sort out nans
        gtvar = groundtruth[:,task_ind]
        prvar = predictions[:,task_ind]
        nan_mask = gtvar != 'nan'
        nan_mask_pr = prvar != 'nan'
        nan_mask = np.logical_and(nan_mask, nan_mask_pr)
        
        gtvar = gtvar[nan_mask]
        prvar = prvar[nan_mask]
        
        ground_truth     = np.squeeze([np.where(gt==list_class_names) for gt in gtvar])
        prediction       = np.squeeze([np.where(pr==list_class_names) for pr in prvar])

        sn_func.estimate_quality_measures(ground_truth=ground_truth,
                                      prediction=prediction,
                                      list_class_names=list(list_class_names),
                                      prefix_plot=taskname,
                                      res_folder_name=logpath,
                                      how_many_training_steps=how_many_training_steps,
                                      bool_MTL=False)


def evaluate_model(pred_gt_path, result_path):
    r""" Evaluates a pre-trained model.
    
    :Arguments\::
        :pred_gt_path (*string*)\::
            Path (without filename) to a "pred_gt.npz" file that was produced by
            the function get_KNN.
        :result_path (*string*)\::
            Path to where the evaluation results will be saved.
            
    :Returns\::
        :groundtruth (*array*)\::
            S-by-V-dimensional array with groundtruth indices, where S is the 
            number of samples and V is the number of variables. 
            This output is only used for the function crossvalidation.
        :predictions (*array*)\::
            S-by-V-dimensional array with prediction indices, where S is the 
            number of samples and V is the number of variables.
            This output is only used for the function crossvalidation.
    
    """
    if not os.path.exists(os.path.join(result_path,r"")): os.makedirs(os.path.join(result_path,r""))
    
    # Load predictions and groundtruth
    vardict = np.load(pred_gt_path+r"/pred_gt.npz", allow_pickle=True)["arr_0"].item()
    predictions = np.asarray(vardict["Predictions"])
    groundtruth = np.asarray(vardict["Groundtruth"])
    label2class_list = np.asarray(vardict["label2class_list"])
    
    for task_ind, classlist in enumerate(label2class_list):
        taskname = classlist[0]
        list_class_names = classlist[1:]

        # sort out nans
        gtvar = groundtruth[:,task_ind]
        prvar = predictions[:,task_ind]

        # uique, counts = np.unique(gtvar, return_counts=True)
        # dictdict = dict(zip(uique, counts))

        nan_mask = gtvar != 'nan'

        # uique, counts = np.unique(nan_mask, return_counts=True)
        # dictdict3 = dict(zip(uique, counts))

        nan_mask_pr = prvar != 'nan'
        nan_mask = np.logical_and(nan_mask, nan_mask_pr)
        
        gtvar = gtvar[nan_mask]
        prvar = prvar[nan_mask]

        # uique, counts = np.unique(nan_mask_pr, return_counts=True)
        # dictdict2 = dict(zip(uique, counts))
        
        ground_truth     = np.squeeze([np.where(gt==list_class_names) for gt in gtvar])
        prediction       = np.squeeze([np.where(pr==list_class_names) for pr in prvar])

        gt_unique, gt_counts = np.unique(ground_truth, return_counts=True)
        if len(gt_unique)  < len(list_class_names):
            print('WARNING: The variable {0!r} has no contribution for '
                  'at least one class in the ground truth. The variable will not be evaluated!'.format(taskname))
            continue
        if any(gt_counts) < 20:
            print('WARNING: The variable {0!r} has a contribution off less than 20 samples for '
                  'at least one class in the ground truth. The evaluation will probably not be representative!'.format(taskname))

        # TODO: Raise warning if less than 20 (?) samples contribute to one of the classes
        # TODO: Raise warning and exclude variable from evaluation if a class has no contribution (due to only nan pred)

        sn_func.estimate_quality_measures(ground_truth=ground_truth,
                                          prediction=prediction,
                                          list_class_names=list(list_class_names),
                                          prefix_plot=taskname,
                                          res_folder_name=result_path,
                                          how_many_training_steps='NaN',
                                          bool_MTL=False)
        
    return groundtruth, predictions


def get_kNN(treepath, master_file_prediction, master_dir_prediction,
            bool_labeled_input, model_path, num_neighbors, savepath):
    r"""Retrieves the k nearest neighbours from a given kdTree.

    :Arguments\::
        :treepath (*string*)\::
            Path (without filename) to a "kdtree.npz"-file that was produced by
            the function build_kDTree.
        :master_file_prediction (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
        :master_dir_prediction (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :bool_labeled_input (*bool*)\::
            A boolean that states wheter labels are available for the input image data (Tue)
            or not(False).
        :model_path (*string*):
            Path (without filename) to a pre-trained network.
        :num_neighbors (*int*)\::
            Number of closest neighbours that are retrieved from a kD-Tree
            during evaluation.
            Default: XXX
        :savepath (*string*):
            Path to where the results will be saved.

    """
    # -> may write extra LUT in build tree with tree ind assigned to feature vector
    # -> what about getting the feature vectors of the kNN
    if bool_labeled_input:
        (pred_top_k,
         labels_target,
         label2class_list) = get_kNN_labeled_input_data(treepath, master_file_prediction, master_dir_prediction,
                                   model_path, num_neighbors, savepath)
    else:
        get_kNN_unlabeled_input_data(treepath, master_file_prediction, master_dir_prediction,
                                     model_path, num_neighbors, savepath)
        pred_top_k = [],
        labels_target = []
        label2class_list = []

    return (pred_top_k, labels_target, label2class_list)


def get_kNN_unlabeled_input_data(treepath, master_file_prediction, master_dir_prediction,
                               model_path, num_neighbors, savepath):
    r"""Retrieves the k nearest neighbours from a given kdTree.

    :Arguments\::
        :treepath (*string*)\::
            Path (without filename) to a "kdtree.npz"-file that was produced by
            the function build_kDTree.
        :master_file_prediction (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
        :master_dir_prediction (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :model_path (*string*):
            Path (without filename) to a pre-trained network.
        :num_neighbors (*int*)\::
            Number of closest neighbours that are retrieved from a kD-Tree
            during evaluation.
            Default: XXX
        :savepath (*string*):
            Path to where the results will be saved.


    """
    # Checks for paths existing: treepath, master file, savepath, model_path
    if not os.path.exists(os.path.join(savepath, r"")): os.makedirs(os.path.join(savepath, r""))

    # Load pre-trained network
    model = ImportGraph(model_path)

    # Load Tree and labels
    tree             = np.load(treepath + r"/kdtree.npz", allow_pickle=True)["arr_0"].item()
    labels_tree      = np.squeeze(tree["Labels"])
    data_dict_train  = tree["DictTrain"]
    relevant_variables = tree["relevant_variables"]
    tree             = tree["Tree"]

    # get images out of collection files
    # and estimate feature vectors
    coll_list = sn_func.master_file_to_collections_list(master_dir_prediction, master_file_prediction)
    features             = []
    all_used_images_test = []
    for collection in coll_list:
        coll_id = open(os.path.join(master_dir_prediction, collection).strip("\n"), 'r')
        for line, rel_im_path in enumerate(coll_id):
            if line == 0: continue
            image_name = os.path.abspath(os.path.join(master_dir_prediction,
                                                      rel_im_path.split('\t')[0].strip("\n")))
            all_used_images_test.append(os.path.relpath(image_name))

            # Get feature vector
            image_data   = tf.gfile.GFile(image_name, 'rb').read()
            features_var = model.run(image_data)
            features.append(features_var)
        coll_id.close()

    # perform query
    dist, ind = tree.query(np.squeeze(features), k=num_neighbors)

    # get predictions out of kNN-classification (mojority vote)
    pred_label_test = []
    pred_names_test = []
    pred_occ_test = []
    pred_top_k = []
    if len(relevant_variables) == 1:
        for k_neighbors in range(np.shape(ind)[0]):
            # list of class labels for all num_neighbors nearest neighbors
            # of the feature vector number k_neighbors in the test set
            temp_pred_list = list(itemgetter(*ind[k_neighbors])(labels_tree))

            # # The most often occuring class label in the predictions
            # # (majority vote)
            # temp_pred_label = stats.mode(temp_pred_list)[0][0]
            # # the name of the estimated class label
            # temp_pred_name = temp_pred_label  # list(data_dict_test.keys())[temp_pred_label]
            # # the number of occurrences of the estimated class label
            # temp_pred_occ = stats.mode(temp_pred_list)[1][0]

            # majority vote without nan-predictions (nan-pred only if all NN nan)
            if temp_pred_list.count('nan') == num_neighbors:
                temp_pred_label = 'nan'
                temp_pred_name = temp_pred_label
                temp_pred_occ = num_neighbors
            else:
                cleaned_pred_list = list(np.asarray(temp_pred_list)[np.asarray(temp_pred_list) != 'nan'])
                temp_pred_label = stats.mode(cleaned_pred_list)[0][0]
                temp_pred_name = temp_pred_label
                temp_pred_occ = stats.mode(cleaned_pred_list)[1][0]

            pred_label_test.append(temp_pred_label)
            pred_names_test.append(temp_pred_name)
            pred_occ_test.append(temp_pred_occ)
            pred_top_k.append(np.asarray(temp_pred_list))
        pred_names_test = np.expand_dims(pred_names_test, axis=1)
    elif len(relevant_variables) > 1:
        for k_neighbors in range(np.shape(ind)[0]):
            # list of class labels for all num_neighbors nearest neighbors
            # in the train set for the feature vector number k_neighbors
            # in the test set
            temp_pred_list = list(itemgetter(*ind[k_neighbors])(labels_tree))

            # # Find the predicted label via majority vote
            # # find the majority for all MTL labels individually
            # temp_pred_label = list(stats.mode(temp_pred_list, axis=0)[0][0])
            # temp_pred_name = temp_pred_label
            #
            # # the number of occurrences of the estimated MTL class labels
            # temp_pred_occ = stats.mode(temp_pred_list)[1][0]

            # majority vote without nan-predictions (nan-pred only if all NN nan)
            temp_pred_label = []
            temp_pred_name = []
            temp_pred_occ = []
            for task_ind in range(len(temp_pred_list[0])):
                task_predictions = np.asarray(temp_pred_list)[:, task_ind]
                if list(task_predictions).count('nan') == num_neighbors:
                    task_pred_label = 'nan'
                    task_pred_name = task_pred_label
                    task_pred_occ = num_neighbors
                else:
                    cleaned_pred_list = list(task_predictions[task_predictions != 'nan'])
                    task_pred_label = stats.mode(cleaned_pred_list)[0][0]
                    task_pred_name = task_pred_label
                    task_pred_occ = stats.mode(cleaned_pred_list)[1][0]
                temp_pred_label.append(task_pred_label)
                temp_pred_name.append(task_pred_name)
                temp_pred_occ.append(task_pred_occ)

            pred_label_test.append(list(np.squeeze(temp_pred_label)))
            pred_names_test.append(list(np.squeeze(temp_pred_name)))
            pred_occ_test.append(np.squeeze(temp_pred_occ))
            pred_top_k.append(np.asarray(temp_pred_list))

    # TODO: D4.6: unlabeled variante of get_top_k_statistics
    #       -> will be the same as the for labeled as no evaluation will be realized

    # Save GT, Prediction, names of k nearest neighbors and their distances to a textfile
    kNN_image_names = []
    # kNN_descriptors = []
    knn_file = open(os.path.abspath(savepath+'/'+'knn_list.txt'), 'w')
    for idx in (range(np.shape(ind)[0])):
        temp_kNN_image_names = []
        # temp_kNN_descriptors = []
        knn_file.write("*******%s*******" % (all_used_images_test[idx]))
        # knn_file.write("\n Groundtruth: \n")
        # for gt in range(np.shape(pred_names_test)[1]):
        #     knn_file.write("%s \t \t" % (np.asarray(all_in_names_test)[idx, gt]))

        knn_file.write("\n Predictions: \n")
        for cur_label in relevant_variables:
            knn_file.write("#%s \t" % cur_label)
        knn_file.write("\n")
        for gt in range(np.shape(pred_names_test)[1]):
            knn_file.write("%s \t \t" % (np.asarray(pred_names_test)[idx, gt]))

        knn_file.write("\n k nearest neighbours: \n")
        knn_file.write("#filename \t #distance ")
        for cur_label in relevant_variables:
            knn_file.write("#%s \t" % cur_label)
        knn_file.write("\n")
        for knn in range(np.shape(ind)[1]):
            knn_file.write("%s \t" % (np.asarray(list(data_dict_train.keys()))[ind[idx][knn]]))
            knn_file.write("%s \t" % (np.asarray(dist)[idx][knn]))

            temp_kNN_image_names.append(np.asarray(list(data_dict_train.keys()))[ind[idx][knn]])
            # temp_image_data = tf.gfile.GFile(np.asarray(list(data_dict_train.keys()))[ind[idx][knn]], 'rb').read()
            # temp_features = model.run(temp_image_data)
            # temp_kNN_descriptors.append(temp_features)

            if len(relevant_variables) > 1:
                for gt in range(np.shape(pred_names_test)[1]):
                    knn_file.write("%s \t" % (np.asarray(labels_tree)[ind[idx][knn], gt]))
            elif len(relevant_variables) == 1:
                knn_file.write("%s \t" % np.asarray(labels_tree)[ind[idx][knn]])
            knn_file.write("\n")
        knn_file.write("\n")

        # temp_kNN_descriptors = [str(list(i)) for i in temp_kNN_descriptors]
        kNN_image_names.append(temp_kNN_image_names)
        # kNN_descriptors.append(temp_kNN_descriptors)
    knn_file.close()

    # select and store data for CSV LUT
    LUT_kNN_dict = {"input_image_name": all_used_images_test,
                    # "image_name": [i.split("\\")[-1].split("__")[2] for i in all_used_images_test[:]],
                    #"input_museum": [i.split("\\")[-1].split("__")[0] for i in all_used_images_test[:]],
                    #"input_kg_object_uri": [i.split("\\")[-1].split("__")[1] for i in all_used_images_test[:]],
                    # "search_descriptor": [str(list(i)) for i in features],
                    "kNN_image_names": kNN_image_names,
                    "kNN_kg_object_uri": [[i.split("\\")[-1].split("__")[1] for i in j[:]] for j in kNN_image_names],
                    "kNN_kD_index": [str(list(i)) for i in ind],
                    "kNN_descriptor_dist": [str(list(i)) for i in dist]}
                    # "kNN_descriptors": kNN_descriptors}
    LUT_df = pd.DataFrame(LUT_kNN_dict)
    LUT_df.to_csv(os.path.join(savepath, "kNN_LUT.csv"), index=False)

    # copy_kNN_images_to_one_folder(LUT_kNN_dict["input_image_name"],
    #                               LUT_kNN_dict["kNN_image_names"],
    #                               LUT_kNN_dict["input_kg_object_uri"],
    #                               LUT_kNN_dict["kNN_descriptor_dist"],
    #                               savepath)


def get_kNN_labeled_input_data(treepath, master_file_prediction, master_dir_prediction,
            model_path, num_neighbors, savepath):
    r"""Retrieves the k nearest neighbours from a given kdTree.
    
    :Arguments\::
        :treepath (*string*)\::
            Path (without filename) to a "kdtree.npz"-file that was produced by
            the function build_kDTree. 
        :master_file_prediction (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
        :master_dir_prediction (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :model_path (*string*):
            Path (without filename) to a pre-trained network.
        :num_neighbors (*int*)\::
            Number of closest neighbours that are retrieved from a kD-Tree
            during evaluation.
            Default: XXX
        :savepath (*string*):
            Path to where the results will be saved.
        
    
    """
    #Checks for paths existing: treepath, master file, savepath, model_path
    if not os.path.exists(os.path.join(savepath,r"")): os.makedirs(os.path.join(savepath,r""))
    
    # Load pre-trained network
    model = ImportGraph(model_path)
    
    # Load Tree and labels
    tree = np.load(treepath+r"/kdtree.npz", allow_pickle=True)["arr_0"].item()
    labels_tree = np.squeeze(tree["Labels"])
    data_dict_train = tree["DictTrain"]
    relevant_variables = tree["relevant_variables"]
    label2class_list = tree["label2class_list"]
    tree = tree["Tree"]

    coll_list = sn_func.master_file_to_collections_list(master_dir_prediction, master_file_prediction)
    coll_dict, data_dict = sn_func.collections_list_MTL_to_image_lists(coll_list,
                                                                  relevant_variables,
                                                                  1,
                                                                  master_dir_prediction,
                                                                  False)

    # get feature vectors and labels, for all samples
    features = []
    labels_target = []
    for image_name in data_dict.keys():
        
        # Get labels, check for incompleteness
        labels_var = []
        for variable in relevant_variables:
            labels_var.append(np.squeeze(data_dict[image_name][variable]))
        
        # Get feature vector
        image_data = tf.gfile.GFile(image_name, 'rb').read()
        features_var = model.run(image_data)
        
        # Save features and labels
        features.append(features_var)
        labels_target.append(labels_var)
    
    # perform query
    dist, ind = tree.query(np.squeeze(features), k=num_neighbors)

    # prediction
    pred_label_test = []
    pred_names_test = []
    pred_occ_test   = []
    pred_top_k      = []
    if len(relevant_variables) == 1:
        for k_neighbors in range(np.shape(ind)[0]):
            # list of class labels for all num_neighbors nearest neighbors
            # of the feature vector number k_neighbors in the test set
            temp_pred_list = list(itemgetter(*ind[k_neighbors])(labels_tree))

            # # The most often occuring class label in the predictions
            # # (majority vote)
            # temp_pred_label = stats.mode(temp_pred_list)[0][0]
            # # the name of the estimated class label
            # temp_pred_name = temp_pred_label # list(data_dict_test.keys())[temp_pred_label]
            # # the number of occurrences of the estimated class label
            # temp_pred_occ = stats.mode(temp_pred_list)[1][0]

            # majority vote without nan-predictions (nan-pred only if all NN nan)
            if temp_pred_list.count('nan') == num_neighbors:
                temp_pred_label = 'nan'
                temp_pred_name = temp_pred_label
                temp_pred_occ = num_neighbors
            else:
                cleaned_pred_list = list(np.asarray(temp_pred_list)[np.asarray(temp_pred_list) != 'nan'])
                temp_pred_label = stats.mode(cleaned_pred_list)[0][0]
                temp_pred_name = temp_pred_label
                temp_pred_occ = stats.mode(cleaned_pred_list)[1][0]

            pred_label_test.append(temp_pred_label)
            pred_names_test.append(temp_pred_name)
            pred_occ_test.append(temp_pred_occ)
            pred_top_k.append(np.asarray(temp_pred_list))
        pred_names_test = np.expand_dims(pred_names_test, axis=1)
    elif len(relevant_variables) > 1:
        for k_neighbors in range(np.shape(ind)[0]):
            # list of class labels for all num_neighbors nearest neighbors
            # in the train set for the feature vector number k_neighbors
            # in the test set
            temp_pred_list  = list(itemgetter(*ind[k_neighbors])(labels_tree))

            # # Find the predicted label via majority vote
            # # find the majority for all MTL labels individually
            # temp_pred_label = list(stats.mode(temp_pred_list, axis=0)[0][0])
            #
            # # ttt = np.asarray(temp_pred_list)
            # # tt1 = ttt[:, 0]
            # temp_pred_name = temp_pred_label
            #
            # # the number of occurrences of the estimated MTL class labels
            # temp_pred_occ   = stats.mode(temp_pred_list)[1][0]

            # majority vote without nan-predictions (nan-pred only if all NN nan)
            temp_pred_label = []
            temp_pred_name = []
            temp_pred_occ = []
            t1 = temp_pred_list[0]
            t2 = len(temp_pred_list[0])
            t3 = np.shape(temp_pred_list[0])
            for task_ind in range(len(temp_pred_list[0])):
                task_predictions = np.asarray(temp_pred_list)[:, task_ind]
                if list(task_predictions).count('nan') == num_neighbors:
                    task_pred_label = 'nan'
                    task_pred_name = task_pred_label
                    task_pred_occ = num_neighbors
                else:
                    cleaned_pred_list = list(task_predictions[task_predictions != 'nan'])
                    task_pred_label = stats.mode(cleaned_pred_list)[0][0]
                    task_pred_name = task_pred_label
                    task_pred_occ = stats.mode(cleaned_pred_list)[1][0]
                temp_pred_label.append(task_pred_label)
                temp_pred_name.append(task_pred_name)
                temp_pred_occ.append(task_pred_occ)

            pred_label_test.append(list(np.squeeze(temp_pred_label)))
            pred_names_test.append(list(np.squeeze(temp_pred_name)))
            pred_occ_test.append(np.squeeze(temp_pred_occ))
            pred_top_k.append(np.asarray(temp_pred_list))

    # TODO: raus für D4.5
    # TODO: D4.6: Aufschlüsseln nach Variable und Klasse
    # TODO: D4.6: neuerliche Evaluierung raus, wird ja zu Standardeval

    # For the estimation of a suitable k
    # get_top_k_statistics(pred_top_k,
    #                      labels_target,
    #                      num_neighbors,
    #                      savepath,
    #                      label2class_list)

    # Save GT, Prediction, names of k nearest neighbors and their distances to a textfile
    all_used_images_test = [os.path.relpath(file) for file in list(data_dict.keys())]
    kNN_image_names = []
    # kNN_descriptors = []
    all_in_names_test = labels_target
    knn_file = open(os.path.abspath(savepath+'/'+'knn_list.txt'), 'w')
    # TODO: D4.6: Güte der Klass. bei Pred. mit angeben (Anzahl occ)
    for idx in (range(np.shape(ind)[0])):
        temp_kNN_image_names = []
        # temp_kNN_descriptors = []
        knn_file.write("*******%s*******" % (all_used_images_test[idx]))
        knn_file.write("\n Groundtruth: \n")
        for cur_label in relevant_variables:
            knn_file.write("#%s \t" % cur_label)
        knn_file.write("\n")
        for gt in range(np.shape(pred_names_test)[1]):
            knn_file.write("%s \t \t" % (np.asarray(all_in_names_test)[idx, gt]))

        knn_file.write("\n Predictions: \n")
        for cur_label in relevant_variables:
            knn_file.write("#%s \t" % cur_label)
        knn_file.write("\n")
        for gt in range(np.shape(pred_names_test)[1]):
            knn_file.write("%s \t \t" % (np.asarray(pred_names_test)[idx, gt]))

        knn_file.write("\n k nearest neighbours: \n")
        knn_file.write("#filename \t #distance ")
        for cur_label in relevant_variables:
            knn_file.write("#%s \t" % cur_label)
        knn_file.write("\n")
        for knn in range(np.shape(ind)[1]):
            knn_file.write("%s \t" % (np.asarray(list(data_dict_train.keys()))[ind[idx][knn]]))
            knn_file.write("%s \t" % (np.asarray(dist)[idx][knn]))

            temp_kNN_image_names.append(np.asarray(list(data_dict_train.keys()))[ind[idx][knn]])
            # temp_image_data = tf.gfile.GFile(np.asarray(list(data_dict_train.keys()))[ind[idx][knn]], 'rb').read()
            # temp_features = model.run(temp_image_data)
            # temp_kNN_descriptors.append(temp_features)

            if len(relevant_variables) > 1:
                for gt in range(np.shape(pred_names_test)[1]):
                    knn_file.write("%s \t" % (np.asarray(labels_tree)[ind[idx][knn], gt]))
            elif len(relevant_variables) == 1:
                knn_file.write("%s \t" % np.asarray(labels_tree)[ind[idx][knn]])
            knn_file.write("\n")
        knn_file.write("\n")

        # temp_kNN_descriptors = [str(list(i)) for i in temp_kNN_descriptors]
        kNN_image_names.append(temp_kNN_image_names)
        # kNN_descriptors.append(temp_kNN_descriptors)
    knn_file.close()

    # Save Prediction and Groundtruth as .npy for evaluation
    pred_gt = {"Groundtruth": np.asarray(all_in_names_test), # indices der Klassen 0, 1, ...
               "Predictions": np.asarray(pred_names_test),   # indices der Klassen 0, 1, ...
               "label2class_list": np.asarray(label2class_list)} # konkrete Namen
    np.savez(savepath+r"/pred_gt.npz", pred_gt)

    # select and store data for CSV LUT
    LUT_kNN_dict = {"input_image_name": all_used_images_test,
                    # "image_name": [i.split("\\")[-1].split("__")[2] for i in all_used_images_test[:]],
                    #"input_museum": [i.split("\\")[-1].split("__")[0] for i in all_used_images_test[:]],
                    #"input_kg_object_uri": [i.split("\\")[-1].split("__")[1] for i in all_used_images_test[:]],
                    # "search_descriptor": [str(list(i)) for i in features],
                    "kNN_image_names": kNN_image_names,
                    "kNN_kg_object_uri": [[i.split("\\")[-1].split("__")[1] for i in j[:]] for j in kNN_image_names],
                    "kNN_kD_index": [str(list(i)) for i in ind],
                    "kNN_descriptor_dist": [str(list(i)) for i in dist]}
                    #"kNN_descriptors": kNN_descriptors}
    LUT_df = pd.DataFrame(LUT_kNN_dict)
    LUT_df.to_csv(os.path.join(savepath, "kNN_LUT.csv"), index=False)

    # copy_kNN_images_to_one_folder(LUT_kNN_dict["input_image_name"],
    #                               LUT_kNN_dict["kNN_image_names"],
    #                               LUT_kNN_dict["input_kg_object_uri"],
    #                               dist,
    #                               savepath)

    return (pred_top_k, labels_target, label2class_list)


def get_top_k_statistics(pred_top_k, ground_truth, num_neighbours,
                         savepath, label2class_list):

    for task_ind, classlist in enumerate(label2class_list):
        taskname = classlist[0]
        task_groundtruth  = np.asarray(ground_truth)[:, task_ind]

        # remove samples where groundtruth is 'nan'
        nan_mask_gt = task_groundtruth != 'nan'
        task_predictions = np.asarray(pred_top_k)[:, :, task_ind]
        task_gt_clean = task_groundtruth[nan_mask_gt]
        tast_prs_clean = task_predictions[nan_mask_gt]

        # 1. average probability of a prediction per variable: #pred_occ/num_neighbours
        count_num_correct_pred = []
        only_labeled_preds = []
        only_labeled_gts = []
        count_all_NN_nan = 0
        count_nan_pred = 0
        for sample_ind in range(len(task_gt_clean)):
            sample_gt = task_gt_clean[sample_ind]
            sample_preds = tast_prs_clean[sample_ind, :]

            # 1. average probability of a prediction per variable: #pred_occ/num_neighbours
            temp_count = list(sample_preds).count(sample_gt)
            count_num_correct_pred.append(temp_count)

            if list(sample_preds).count('nan') == num_neighbours:
                # 2.1 amount of all num_neighbours 'nan' per variable: #all_nan/len(groundtruth)
                count_all_NN_nan += 1
            else:
                # 2.2 kNN-classification measures for majority without 'nan' in num_neighbours:
                # estimate_quality_measures
                labeled_preds = sample_preds[sample_preds != 'nan']
                resulting_pred = stats.mode(labeled_preds)[0][0]
                only_labeled_preds.append(resulting_pred)
                only_labeled_gts.append(sample_gt)

            # 3. kNN-classification (if nan:2, else:2 -> else), count amount 'nan'-predictions
            if list(sample_preds).count('nan') >= np.floor(num_neighbours/2):
                count_nan_pred += 1

        # to 1. average probability of a prediction per variable: #pred_occ/num_neighbours
        average_precision = np.asarray(count_num_correct_pred) / num_neighbours
        mean_average_precision = np.mean(average_precision)

        # to 2.1 amount of all num_neighbours 'nan' per variable: #all_nan/len(groundtruth)
        ratio_all_nan = count_all_NN_nan/len(task_gt_clean)

        # to 2.2 kNN-classification measures for majority ohne 'nan' in num_neighbours: estimate_quality_measures
        # contributions are not correctly written by now, because names are provided, but needs indices
        list_class_names = classlist[1:]
        sn_func.estimate_quality_measures(ground_truth=only_labeled_gts,
                                          prediction=only_labeled_preds,
                                          list_class_names=list(list_class_names),
                                          prefix_plot=taskname + '_no_nans_in_kNN',
                                          res_folder_name=os.path.join(savepath, 'top_k_statistics'),
                                          how_many_training_steps='NaN',
                                          bool_MTL=False)

        # merge all without 2.2
        text_file_kNN_result = open(os.path.join(savepath, 'top_k_statistics') \
                                    + '/' + taskname + '_kNN_statistics.txt', 'w')
        text_file_kNN_result.write('Performed {:.0f} nearest neighbour search.'.format(num_neighbours))

        text_file_kNN_result.write('\n\nThe average probability of the predictions'
                                   ' (correct predictions/num_neighbours) averaged over all'
                                   ' {:.0f} samples is:\n{:.2f}%'.format(len(average_precision), mean_average_precision*100))

        text_file_kNN_result.write('\n\nThe occurrence of {:.0f} times nan in the '
                                   '{:.0f} predictions (belonging to groudtruths with labels)'
                                   ' is:\n{:.2f}%'.format (num_neighbours, len(task_gt_clean), ratio_all_nan*100))

        text_file_kNN_result.write('\n\nThe amount of nan-predictions within majority vote is:\n{:.2f}%'.format(
                                   count_nan_pred/len(task_gt_clean) * 100))
        text_file_kNN_result.close()


def copy_kNN_images_to_one_folder(input_image_names, kNN_image_names, input_kg_object_uris,
                                  kNN_descriptor_dist, savepath):
    """
    """
    main_dir = os.path.join(savepath, 'similar_image_folder')
    for input_image, input_uri, kNNs, kNN_dist in zip(input_image_names, input_kg_object_uris,
                                                      kNN_image_names, kNN_descriptor_dist):
        image_dir = os.path.join(main_dir, input_uri)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # copy the input image to this folder
        copy(input_image.replace("\\img\\", "\\img_unscaled\\"), image_dir)

        # copy similar images to subfolder
        current_subdir = os.path.join(image_dir, str(len(kNNs)) + "_most_similar_images")
        if not os.path.exists(current_subdir):
            os.makedirs(current_subdir)
        for NN_ind, NN in enumerate(kNNs):
            temp1 = NN.replace("\\img\\", "\\img_unscaled\\")
            temp2 = os.path.abspath(os.path.join(current_subdir, "{:.4f}_".format(kNN_dist[NN_ind]) + os.path.basename(NN)))
            copy(temp1, temp2)


def build_kDTree(model_path, master_file_tree, master_dir_tree, 
                 relevant_variables, savepath):
    r"""Builds a kD-Tree using a pre-trained network.
    
    :Arguments\::
        :model_path (*string*):
            Path (without filename) to a folder containing a pre-trained network. Only one pretrained
            model should be stored in that model folder. It refers to the logpath in the
            function 'train_model_parameter'.
        :master_file_tree (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            All samples from all stated collection files will be fed into the tree.
        :master_dir_tree (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :relevant_variables (*list*)\::
            A list containing the collection.txt's header terms of the labels
            to be considered. The terms must not contain blank spaces; use '_'
            instead of blank spaces!
            - Example (string in control file): #timespan, #place
            - Example (according list in code): [timespan, place]
            Default: XXX
        :savepath (*string*):
            Path to where the kD-tree will be saved.
    
    """
    #Checks for paths existing: model_path, master file, savepath
    if not os.path.exists(os.path.join(savepath,r"")): os.makedirs(os.path.join(savepath,r""))
    
    # Load pre-trained network
    model = ImportGraph(model_path)
    
    # load samples from which the tree will be created
    coll_list = sn_func.master_file_to_collections_list(master_dir_tree, master_file_tree)
    coll_dict, data_dict = sn_func.collections_list_MTL_to_image_lists(coll_list,
                                                                  relevant_variables,
                                                                  1,
                                                                  master_dir_tree,
                                                                  False)

    cur_dir = os.path.abspath(os.getcwd())
    rel_paths = [os.path.relpath(i, cur_dir) for i in data_dict.keys()]
    tree_data_dict = data_dict
    for (old_key, new_key) in zip(list(data_dict.keys()), rel_paths):
        tree_data_dict[new_key] = tree_data_dict.pop(old_key)
        
    # lists begin with task name, followed by all corresponding classes
    label2class_list = []
    for label_key in relevant_variables:
        varname = np.asarray(label_key)
        varlist = np.asarray(list(coll_dict[label_key].keys()))
        varlist = np.insert(varlist,0,varname)
        label2class_list.append(varlist)
    
    # get feature vectors and labels, for all samples
    features_all = []
    labels_all = []
    print("Estimating descriptors:")
    for image_name in tqdm(data_dict.keys()):

        # Get labels, check for incompleteness
        labels_var = []
        for variable in relevant_variables:
            labels_var.append(data_dict[image_name][variable])

        # Get feature vector
        image_data = tf.gfile.GFile(image_name, 'rb').read()
        features = model.run(image_data)

        # Save features and labels
        features_all.append(features)
        labels_all.append(labels_var)
        
    # build tree
    tree = KDTree(np.squeeze(features_all), leaf_size=2)
    
    # export tree and labels
    tree_with_labels = {"Tree": tree,
                        "Labels": labels_all,
                        "DictTrain": tree_data_dict,
                        "relevant_variables": relevant_variables,
                        "label2class_list": label2class_list
                        }
    np.savez(savepath+r"/kdtree.npz", tree_with_labels)
        
    #return tree_with_labels


class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        """Creates an object of class ImportGraph
        
        :Arguments:
          :loc:
              The absolute path to the storage location of the trained graph
            including the name of the graph.
          :output_name:
            The name of the output classification layer in the retrained graph.
            It has to be the same name as it was given in the training.
        :task_list:
            Names of the tasks to be considered for the classification. The
            wanted tasks have to be contained in the label_file, i.e. they must
            have been considered during training, too. Task names should begin
            with a # and be separated by commas, e.g. '#timespan, #place'
    
        :Returns:
            Object of class ImportGraph
        """
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + CHECKPOINT_NAME + '.meta',
                                               clear_devices=True)
            """EXPERIMENTELL"""
            init = tf.global_variables_initializer()
            self.sess.run(init)
#            self.sess.run(tf.local_variables_initializer())
            """EXPERIMENTELL"""
            saver.restore(self.sess, loc + CHECKPOINT_NAME)
            #self.output_features = self.graph.get_operation_by_name('CustomLayers/output_features').outputs[0]
            self.output_features = self.graph.get_operation_by_name('l2_normalize').outputs[0]



    def run(self, data):
        """ Running the activation operation previously imported.
        
        :Arguments:
            :data:
                The image data, i.e. the output from read_tensor_from_image_file.
                
        :Returns:
            :output:
                The result of the specified layer (output_name).
        """
        # The 'x' corresponds to name of input placeholder
        feed_dict_raw={"DecodeJPGInput:0": data}
        decoded_img_op = self.graph.get_operation_by_name('Squeeze').outputs[0]
        decoded_img = self.sess.run(decoded_img_op, feed_dict=feed_dict_raw)
        decoded_img = np.expand_dims(np.asarray(decoded_img), 0)
        
        feed_dict_decoded={"ModuleLayers/input_img:0": decoded_img}
        output = self.sess.run(self.output_features, feed_dict=feed_dict_decoded)
        output = np.squeeze(output)
        return output


def read_configfile_create_dataset(configfile):
    """ Reads the control file for the function createDataset_config.
    
     :Arguments\::
        :configfile (*string*)\::
            This variable is a string and contains the path (including filename)
            of the configuration file. All relevant information for setting up
            the data set is in this file.
            
    :Returns\::
        :csvfile (*string*)\::
            Filename of the .csv file that represents all data used
            for training and testing the classifier.
            This file has to exist in the sub-folder /samples/.
        :imgsavepath (*string*)\::
            Path to where the images will be downloaded. This path has
            to be relative to the main software folder.
        :minnumsaples (*int*)\::
            Minimum number of samples for each class. Classes with fewer
            occurences will be ignored and set to unknown.
        :retaincollections (*list*)\::
            List of strings that defines the museums/collections that
            are be used. Data from museums/collections
            not stated in this list will be omitted.
            
    """""
    control_id = open(configfile, 'r', encoding='utf-8')
    for variable in control_id:
        if variable.split(';')[0] == 'csvfile':
            csvfile = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'imgsavepath':
            imgsavepath = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'minnumsamples':
            minnumsamples = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'retaincollections':
            retaincollections = variable.split(';')[1]\
                    .replace(' ', '').replace('\n', '')\
                    .replace('\t', '').split(',')
    return(csvfile,
           imgsavepath,
           minnumsamples,
           retaincollections)


def read_configfile_crossvalidation(configfile):
    r""" Reads the control file for the function crossvalidation.
    
    :Arguments\::
        :configfile (*string*)\::
            Path (including filename) to the configuration file defining the
            parameters for the function crossvalidation.
    :Returns\::
        :master_file (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
        :master_dir (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :logpath (*string*):
            The path where all summarized training information and the trained
            network will be stored.
        :train_batch_size (*int*)\::
            This variable defines how many images shall be used for
            the classifier's training for one training step.
            Default: XXX.
        :how_many_training_steps (*int*)\::
            Number of training iterations.
        :learning_rate (*float*)\::
            Specifies the learning rate of the Optimizer.
            Default: XXX.
        :add_fc (*array of int*)\::
            The number of fully connected layers that shall be trained on top
            of the tensorflow hub Module is equal to the number of entries in
            the array. Each entry is an int specifying the number of nodes in
            the individual fully connected layers. If [1000, 100] is given, two
            fc layers will be added, where the first has 1000 nodes and the
            second has 100 nodes. If no layers should be added, an empty array
            '[]' has to be given.
            Default: XXX.
        :hub_num_retrain (*int*)\::
            The number of layers from the
            tensorflow hub Module that shall be retrained. 
            Default: XXX.
        :aug_dict (*dict*)\::
            A dictionary specifying which types of data augmentations shall 
            be applied during training. A list of available augmentations can be
            found in the documentation of the SILKNOW WP4 Library.
            Default: XXX
        :optimizer_ind (*string*)\::
            The optimizer that shall be used during the training procedure of
            the siamese network. Possible options:
                'Adagrad' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
                'Adam' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
                'GradientDescent' (cf. https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)
            Default: XXX
        :loss_ind (*string*)\::
            The loss function that shall be utilized to estimate the network
            performance during training. Possible options:
                'contrastive'       (cf. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964)
                'soft_contrastive' (own development)
                'contrastive_thres' (own development)
                'triplet_loss'      (cf. https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
                'soft_triplet'  (own development; ~triplet_multi)
                'triplet_thresh'    (own development) 
            Default: XXX
        :relevant_variables (*list*)\::
            A list containing the collection.txt's header terms of the labels
            to be considered. The terms must not contain blank spaces; use '_'
            instead of blank spaces!
            Example (string in control file): #timespan, #place  
            Example (according list in code): [timespan, place]
            Default: XXX
        :similarity_thresh (*float*)\::
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
            Default: XXX
        :label_weights (*list*)\::
            Weights that express the importance of the individual labels for
            the similarity estimation. Have to be positive numbers. Will be
            normalized so that the sum is 1.
            The list has to be as long as "relevant_variables".
            Default: XXX
        :how_often_validation (*int*)\::
            Number of training iterations between validations.
            Default: XXX
        :val_percentage (*int*)\::
            Percentage of training data that will be used for validation.
            Default: XXX
        :num_neighbors (*int*)\::
            Number of closest neighbours that are retrieved from a kD-Tree
            during evaluation.
            Default: XXX
         
    """
    control_id = open(configfile, 'r',encoding='utf-8')
    bool_data_aug = True
    aug_set_dict = {}
    # random_shear = None
    # random_brightness = None
    # random_rotation = None
    # random_contrast = None
    # random_hue = None
    # random_saturation = None
    for variable in control_id:
        if variable.split(';')[0] == 'master_file_name':
            master_file = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_dir':
            master_dir = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'logpath':
            logpath = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'train_batch_size':
            train_batch_size = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'how_many_training_steps':
            how_many_training_steps = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'learning_rate':
            learning_rate = np.float(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'val_percentage':
            val_percentage = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'tfhub_module':
            tfhub_module = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'add_fc':
            if len(variable.split('[')[1].split(']')[0]) > 0:
                add_fc = list(map(int,
                        variable.split('[')[1].split(']')[0].split(',')))
            else:
                add_fc = []
        if variable.split(';')[0] == 'hub_num_retrain':
            hub_num_retrain = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'bool_data_aug':
            bool_data_aug = variable.split(';')[1].strip()
            if bool_data_aug == 'True':
                bool_data_aug = True
            else:
                bool_data_aug = False
        # if bool_data_aug:
        #     if variable.split(';')[0] == 'flip_left_right':
        #         flip_left_right = variable.split(';')[1].strip()
        #         if flip_left_right == 'True':
        #             flip_left_right = True
        #         else:
        #             flip_left_right = False
        #     if variable.split(';')[0] == 'flip_up_down':
        #         flip_up_down = variable.split(';')[1].strip()
        #         if flip_up_down == 'True':
        #             flip_up_down = True
        #         else:
        #             flip_up_down = False
        #     if variable.split(';')[0] == 'random_shear':
        #         random_shear = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'random_brightness':
        #         random_brightness = int(variable.split(';')[1].strip())
        #     if variable.split(';')[0] == 'random_crop':
        #         random_crop = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'random_rotation':
        #         random_rotation = float(variable.split(';')[1].strip())*math.pi/180
        #     if variable.split(';')[0] == 'random_contrast':
        #         random_contrast = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'gaussian_noise':
        #         gaussian_noise = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'random_hue':
        #         random_hue = float(variable.split(';')[1].strip())
        #     if variable.split(';')[0] == 'random_saturation':
        #         random_saturation = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'random_rotation90':
        #         random_rotation90 = variable.split(';')[1].strip()
        #         if random_rotation90 == 'True':
        #             random_rotation90 = True
        #         else:
        #             random_rotation90 = False

        # Augmentation
        if variable.split(';')[0] == 'flip_left_right':
            flip_left_right = variable.split(';')[1].strip()
            if flip_left_right == 'True':
                flip_left_right = True
            else:
                flip_left_right = False
            aug_set_dict['flip_left_right'] = flip_left_right
        if variable.split(';')[0] == 'flip_up_down':
            flip_up_down = variable.split(';')[1].strip()
            if flip_up_down == 'True':
                flip_up_down = True
            else:
                flip_up_down = False
            aug_set_dict['flip_up_down'] = flip_up_down
        if variable.split(';')[0] == 'random_shear':
            random_shear = list(map(float,
                                    variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_shear'] = random_shear
        if variable.split(';')[0] == 'random_brightness':
            random_brightness = int(variable.split(';')[1].strip())
            aug_set_dict['random_brightness'] = random_brightness
        if variable.split(';')[0] == 'random_crop':
            random_crop = list(map(float,
                                   variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_crop'] = random_crop
        if variable.split(';')[0] == 'random_rotation':
            random_rotation = float(variable.split(';')[1].strip()) * math.pi / 180
            aug_set_dict['random_rotation'] = random_rotation
        if variable.split(';')[0] == 'random_contrast':
            random_contrast = list(map(float,
                                       variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_contrast'] = random_contrast
        if variable.split(';')[0] == 'random_hue':
            random_hue = float(variable.split(';')[1].strip())
            aug_set_dict['random_hue'] = random_hue
        if variable.split(';')[0] == 'random_saturation':
            random_saturation = list(map(float,
                                         variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_saturation'] = random_saturation
        if variable.split(';')[0] == 'random_rotation90':
            random_rotation90 = variable.split(';')[1].strip()
            if random_rotation90 == 'True':
                random_rotation90 = True
            else:
                random_rotation90 = False
            aug_set_dict['random_rotation90'] = random_rotation90
        if variable.split(';')[0] == 'gaussian_noise':
            gaussian_noise = float(variable.split(';')[1].strip())
            aug_set_dict['gaussian_noise'] = gaussian_noise


        if variable.split(';')[0] == 'optimizer_ind':        
            optimizer_ind = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'loss_ind':        
            loss_ind = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'relevant_variables':
            relevant_variables = variable.split(';')[1].replace(',', '')\
                    .replace(' ', '').replace('\n', '')\
                    .replace('\t', '').split('#')[1:]
        if variable.split(';')[0] == 'similarity_thresh':
            similarity_thresh = np.float(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'label_weights':
            label_weights = list(map(float,
                        variable.split('[')[1].split(']')[0].split(',')))
            # normalize label weights
            if label_weights:
                label_weights= label_weights/np.sum(label_weights)
            else:
                label_weights = np.ones(len(relevant_variables))/len(relevant_variables)
        if variable.split(';')[0] == 'how_often_validation':
            how_often_validation = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'val_percentage':
            val_percentage = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'num_neighbors':
            num_neighbors = np.int(variable.split(';')[1].strip())
            
    control_id.close()
    
    # if bool_data_aug:
    #     aug_dict = {"flip_left_right": flip_left_right,
    #                 "flip_up_down": flip_up_down,
    #                 "random_shear": random_shear,
    #                 "random_brightness": random_brightness,
    #                 "random_crop": random_crop,
    #                 "random_rotation": random_rotation,
    #                 "random_contrast": random_contrast,
    #                 "random_hue": random_hue,
    #                 "random_saturation": random_saturation,
    #                 "random_rotation90": random_rotation90,
    #                 "gaussian_noise": gaussian_noise}
    # else:
    #     aug_dict = {}

    similarity_thresh = 1
    label_weights = list(np.ones(len(relevant_variables)))

    return(master_file, master_dir, logpath, 
           train_batch_size, how_many_training_steps, learning_rate, 
           add_fc, hub_num_retrain, aug_set_dict, loss_ind,
           relevant_variables, similarity_thresh, label_weights, 
           how_often_validation, val_percentage, num_neighbors)


def read_configfile_evaluate_model(configfile): 
    r""" Reads the control file for the function evaluate_model.
    
    :Arguments\::
        :configfile (*string*)\::
            Path (including filename) to the configuration file defining the
            parameters for the function evaluate_model.

    :Returns\::
        :pred_gt_path (*string*)\::
            Path (without filename) to a "pred_gt.npz" file that was produced by
            the function get_KNN.
        :result_path (*string*)\::
            Path to where the evaluation results will be saved.
         
    """
    control_id = open(configfile, 'r',encoding='utf-8')
    for variable in control_id:
        if variable.split(';')[0] == 'pred_gt_path':
            pred_gt_path = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'result_path':
            result_path = variable.split(';')[1].strip()
            
    return pred_gt_path, result_path

    
def read_configfile_get_kNN(configfile):
    r"""Reads the control file for the function get_kNN.
    
    :Arguments\::
        :configfile (*string*)\::
            Path (including filename) to the configuration file defining the
            parameters for the function get_kNN.
            
    :Returns\::
        :treepath (*string*)\::
            Path (without filename) to a "kdtree.npz"-file that was produced by
            the function build_kDTree. 
        :master_file_prediction (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
        :master_dir_prediction (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :bool_labeled_input (*bool*)\::
            A boolean that states wheter labels are available for the input image data (Tue)
            or not(False).
        :model_path (*string*):
            Path (without filename) to a pre-trained network.
        :bool_labeled_input (*bool*)\::
            A boolean that states wheter labels are available for the input image data (Tue)
            or not(False).
        :num_neighbors (*int*)\::
            Number of closest neighbours that are retrieved from a kD-Tree
            during evaluation.
            Default: XXX
        :savepath (*string*):
            Path to where the results will be saved.
    """
    control_id = open(configfile, 'r',encoding='utf-8')
    for variable in control_id:
        if variable.split(';')[0] == 'treepath':
            treepath = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_file_prediction':
            master_file_prediction = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_dir_prediction':
            master_dir_prediction = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'model_path':
            model_path = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'bool_labeled_input':
            bool_labeled_input = variable.split(';')[1].strip()
            if bool_labeled_input == 'True':
                bool_labeled_input = True
            else:
                bool_labeled_input = False
        if variable.split(';')[0] == 'num_neighbors':
            num_neighbors = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'savepath':
            savepath = variable.split(';')[1].strip()
            
    return (treepath, master_file_prediction, master_dir_prediction, bool_labeled_input,
            model_path, num_neighbors, savepath)


def read_configfile_build_kDTree(configfile):
    r"""Reads the control file for the function build_kDTree.
    
    :Arguments\::
        :configfile (*string*)\::
            Path (including filename) to the configuration file defining the
            parameters for the function build_kDTree.
            
    :Returns\::
        :model_path (*string*):
            Path (without filename) to a pre-trained network.
        :master_file_tree (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
        :master_dir_tree (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :relevant_variables (*list*)\::
            A list containing the collection.txt's header terms of the labels
            to be considered. The terms must not contain blank spaces; use '_'
            instead of blank spaces!
            Example (string in control file): #timespan, #place  
            Example (according list in code): [timespan, place]
            Default: XXX
        :savepath (*string*):
            Path to where the results will be saved.
    """
    
    control_id = open(configfile, 'r',encoding='utf-8')
    for variable in control_id:
        if variable.split(';')[0] == 'model_path':
            model_path = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_file_tree':
            master_file_tree = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_dir_tree':
            master_dir_tree = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'relevant_variables':
            relevant_variables = variable.split(';')[1].replace(',', '')\
                    .replace(' ', '').replace('\n', '')\
                    .replace('\t', '').split('#')[1:]
        if variable.split(';')[0] == 'savepath':
            savepath = variable.split(';')[1].strip()
            
    return model_path, master_file_tree, master_dir_tree, relevant_variables, savepath


def read_configfile_train_model(configfile):
    r"""Reads the control file for the function train_model.
    
    :Arguments\::
        :configfile (*string*)\::
            Path (including filename) to the configuration file defining the
            parameters for the function train_model.
            
    :Returns\::
        :master_file (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            All samples from all stated collection files will be used for training and,
            possibly, validation.
        :master_dir (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :logpath (*string*):
            The path where all summarized training information and the trained
            network will be stored.
        :train_batch_size (*int*)\::
            This variable defines how many images shall be used for
            the classifier's training for one training step.
            Default: XXX.
        :how_many_training_steps (*int*)\::
            Number of training iterations.
        :learning_rate (*float*)\::
            Specifies the learning rate of the Optimizer.
            Default: XXX.
        :add_fc (*array of int*)\::
            The number of fully connected layers that shall be trained on top
            of the tensorflow hub Module is equal to the number of entries in
            the array. Each entry is an int specifying the number of nodes in
            the individual fully connected layers. If [1000, 100] is given, two
            fc layers will be added, where the first has 1000 nodes and the
            second has 100 nodes. If no layers should be added, an empty array
            '[]' has to be given.
            Default: XXX.
        :hub_num_retrain (*int*)\::
            The number of layers from the
            tensorflow hub Module that shall be retrained. 
            Default: XXX.
        :optimizer_ind (*string*)\::
            The optimizer that shall be used during the training procedure of
            the siamese network. Possible options:
                'Adagrad' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
                'Adam' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
                'GradientDescent' (cf. https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)
            Default: XXX
        :loss_ind (*string*)\::
            The loss function that shall be utilized to estimate the network
            performance during training. Possible options:
                'contrastive'       (cf. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964)
                'soft_contrastive' (own development)
                'contrastive_thres' (own development)
                'triplet_loss'      (cf. https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
                'soft_triplet'  (own development; ~triplet_multi)
                'triplet_thresh'    (own development) 
            Default: XXX
        :relevant_variables (*list*)\::
            A list containing the collection.txt's header terms of the labels
            to be considered. The terms must not contain blank spaces; use '_'
            instead of blank spaces!
            Example (string in control file): #timespan, #place  
            Example (according list in code): [timespan, place]
            Default: XXX
        :similarity_thresh (*float*)\::
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
            Default: XXX
        :label_weights (*list*)\::
            Weights that express the importance of the individual labels for
            the similarity estimation. Have to be positive numbers. Will be
            normalized so that the sum is 1.
            The list has to be as long as "relevant_variables".
            Default: XXX
        :how_often_validation (*int*)\::
            Number of training iterations between validations.
            Default: XXX
        :val_percentage (*int*)\::
            Percentage of training data that will be used for validation.
            Default: XXX
    """
    bool_data_aug     = True
    # random_shear = None
    # random_brightness = None
    # random_rotation = None
    # random_contrast = None
    # random_hue = None
    # random_saturation = None
    loss_ind          = 'NaN'
    relevant_variables   = []
    similarity_thresh = 1.0
    label_weights     = []
    control_id = open(configfile, 'r',encoding='utf-8')
    aug_set_dict = {}
    for variable in control_id:
        if variable.split(';')[0] == 'master_file_name':
            master_file_name = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'master_dir':
            master_dir = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'train_batch_size':
            train_batch_size = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'how_many_training_steps':
            how_many_training_steps = np.int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'learning_rate':
            learning_rate = np.float(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'val_percentage':
            val_percentage = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'tfhub_module':
            tfhub_module = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'logpath':
            logpath = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'add_fc':
            if len(variable.split('[')[1].split(']')[0]) > 0:
                add_fc = list(map(int,
                        variable.split('[')[1].split(']')[0].split(',')))
            else:
                add_fc = []
        if variable.split(';')[0] == 'hub_num_retrain':
            hub_num_retrain = int(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'bool_data_aug':
            bool_data_aug = variable.split(';')[1].strip()
            if bool_data_aug == 'True':
                bool_data_aug = True
            else:
                bool_data_aug = False
        # if bool_data_aug:
        #     if variable.split(';')[0] == 'flip_left_right':
        #         flip_left_right = variable.split(';')[1].strip()
        #         if flip_left_right == 'True':
        #             flip_left_right = True
        #         else:
        #             flip_left_right = False
        #     if variable.split(';')[0] == 'flip_up_down':
        #         flip_up_down = variable.split(';')[1].strip()
        #         if flip_up_down == 'True':
        #             flip_up_down = True
        #         else:
        #             flip_up_down = False
        #     if variable.split(';')[0] == 'random_shear':
        #         random_shear = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'random_brightness':
        #         random_brightness = int(variable.split(';')[1].strip())
        #     if variable.split(';')[0] == 'random_crop':
        #         random_crop = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'random_rotation':
        #         random_rotation = float(variable.split(';')[1].strip())*math.pi/180
        #     if variable.split(';')[0] == 'random_contrast':
        #         random_contrast = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'gaussian_noise':
        #         gaussian_noise = np.float(variable.split(';')[1].strip())
        #     if variable.split(';')[0] == 'random_hue':
        #         random_hue = float(variable.split(';')[1].strip())
        #     if variable.split(';')[0] == 'random_saturation':
        #         random_saturation = list(map(float,
        #                     variable.split('[')[1].split(']')[0].split(',')))
        #     if variable.split(';')[0] == 'random_rotation90':
        #         random_rotation90 = variable.split(';')[1].strip()
        #         if random_rotation90 == 'True':
        #             random_rotation90 = True
        #         else:
        #             random_rotation90 = False

        # Augmentation
        if variable.split(';')[0] == 'flip_left_right':
            flip_left_right = variable.split(';')[1].strip()
            if flip_left_right == 'True':
                flip_left_right = True
            else:
                flip_left_right = False
            aug_set_dict['flip_left_right'] = flip_left_right
        if variable.split(';')[0] == 'flip_up_down':
            flip_up_down = variable.split(';')[1].strip()
            if flip_up_down == 'True':
                flip_up_down = True
            else:
                flip_up_down = False
            aug_set_dict['flip_up_down'] = flip_up_down
        if variable.split(';')[0] == 'random_shear':
            random_shear = list(map(float,
                                    variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_shear'] = random_shear
        if variable.split(';')[0] == 'random_brightness':
            random_brightness = int(variable.split(';')[1].strip())
            aug_set_dict['random_brightness'] = random_brightness
        if variable.split(';')[0] == 'random_crop':
            random_crop = list(map(float,
                                   variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_crop'] = random_crop
        if variable.split(';')[0] == 'random_rotation':
            random_rotation = float(variable.split(';')[1].strip()) * math.pi / 180
            aug_set_dict['random_rotation'] = random_rotation
        if variable.split(';')[0] == 'random_contrast':
            random_contrast = list(map(float,
                                       variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_contrast'] = random_contrast
        if variable.split(';')[0] == 'random_hue':
            random_hue = float(variable.split(';')[1].strip())
            aug_set_dict['random_hue'] = random_hue
        if variable.split(';')[0] == 'random_saturation':
            random_saturation = list(map(float,
                                         variable.split('[')[1].split(']')[0].split(',')))
            aug_set_dict['random_saturation'] = random_saturation
        if variable.split(';')[0] == 'random_rotation90':
            random_rotation90 = variable.split(';')[1].strip()
            if random_rotation90 == 'True':
                random_rotation90 = True
            else:
                random_rotation90 = False
            aug_set_dict['random_rotation90'] = random_rotation90
        if variable.split(';')[0] == 'gaussian_noise':
            gaussian_noise = float(variable.split(';')[1].strip())
            aug_set_dict['gaussian_noise'] = gaussian_noise



        if variable.split(';')[0] == 'optimizer_ind':        
            optimizer_ind = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'loss_ind':        
            loss_ind = variable.split(';')[1].strip()
        if variable.split(';')[0] == 'relevant_variables':
            relevant_variables = variable.split(';')[1].replace(',', '')\
                    .replace(' ', '').replace('\n', '')\
                    .replace('\t', '').split('#')[1:]
        if variable.split(';')[0] == 'similarity_thresh':
            similarity_thresh = np.float(variable.split(';')[1].strip())
        if variable.split(';')[0] == 'how_often_validation':
            how_often_validation = np.int(variable.split(';')[1].strip())
            
    control_id.close()
    
    # if bool_data_aug:
    #     aug_dict = {"flip_left_right": flip_left_right,
    #                 "flip_up_down": flip_up_down,
    #                 "random_shear": random_shear,
    #                 "random_brightness": random_brightness,
    #                 "random_crop": random_crop,
    #                 "random_rotation": random_rotation,
    #                 "random_contrast": random_contrast,
    #                 "random_hue": random_hue,
    #                 "random_saturation": random_saturation,
    #                 "random_rotation90": random_rotation90,
    #                 "gaussian_noise": gaussian_noise}
    # else:
    #     aug_dict = {}


    similarity_thresh = 1
    label_weights = list(np.ones(len(relevant_variables)))

    if loss_ind in ['soft_triplet', 'triplet_thresh','soft_triplet_incomp_loss']:
        if len(label_weights)>0 and len(label_weights) != len(relevant_variables):
            print('"label_weights" has to have the same length as\
                   "relevant_variables" or has to be empty!')            
            sys.exit()

    return(master_file_name, master_dir, logpath,
           train_batch_size, how_many_training_steps, learning_rate, 
           add_fc, hub_num_retrain, aug_set_dict, loss_ind,
           relevant_variables, similarity_thresh, label_weights, how_often_validation, val_percentage)


def create_batch(sess, batch_size, data_dict, used_images, jpeg_data_tensor,
                 in_img_tensor, label2class_dict, loss_ind, evaluation=False):
    r"""Generates a batch.
    
    :Arguments\::
        :sess (*tf.Session*)\::
            Current active TensorFlow Session.
        :batch_size (*int*)\::
            This variable is an int and says how many images shall be used in a
            batch.
            
            batch_size > 0:
                The given number von samples will randomly picked out of all
                given images in the data_dict.
            batch_size = -1:
                All given images in data_dict will be used for the batch.
        :data_dict (*dictionary*)\::
            If loss_ind in ["triplet_loss", "contrastive"]:
                A dictionary containing the assignments of images to the
                classes. It's data_dict[class_label][img1, ..., imgN], where
                "img" is the absolute path to the image including the image
                name.
            If loss_ind in ['soft_triplet', 'triplet_thresh',
            'soft_contrastive']:
                A dictionary containing the assignments of images to the
                class labels. It's data_dict[image][task][class], where "image"
                is the absolute path to the image including the image name.
        :used_images (*list*)\::
            Contains all image names of images that are already used and thus
            won't be considered for the current batch creation.
        :jpeg_data_tensor (*tensor*)\::
            The tensor to feed the data (individual images) to.
        :in_img_tensor (*tensor*)\::
            The tensor for data (one image) that is already pre-processed and
            thus can be provided to the network. It has the correct input
            dimensions for the net.
        :label2class_dict (*dictionary*)\::
            Contains the assigments of the classes to the individual tasks. It
            has the structure label2class_dict[task][class1, ..., classK].
        :loss_ind (*string*)\::
            The loss function that shall be utilized to estimate the network
            performance during training. Possible options:
                'contrastive'       (cf. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964)
                'soft_contrastive' (own development)
                'contrastive_thres' (own development)
                'magnet_loss'       (cf. https://arxiv.org/pdf/1511.05939.pdf)
                'triplet_loss'      (cf. https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
                'soft_triplet'  (own development; ~triplet_multi)
                'triplet_thres'     (own development)   
        :frac_pos_samples (*float*)\::
            The percentage of sample pairs belonging to the same class (matching
            is positive, similar=1). Accordingly, 1-frac_pos_samples sample
            pairs contain samples of different classes (similar=0).
    
    :Returns\::
        :used_images (*list*)\::
            The input list of used_images expanded by the images that are used
            to build the current batch.
        :batch_in_img (*list*)\::
            A list of images that build the current batch.
            If loss_ind in ['triplet_loss', 'soft_triplet', 'triplet_thresh']:
                (len(batch_in_label) = batch_size)
            If loss_ind in ['contrastive']:
                The first half of the list is for the left and the second half
                for the right network input.
                (len(batch_in_label) = 2 x batch_size)
        :batch_in_label (*list (of lists)*)\::
            A list of class labels (indexes) according to the images in the
            current batch.
            If loss_ind = "triplet_loss":
                One class label per image. (len(batch_in_label) = batch_size)
            If loss_ind in ['soft_triplet', 'triplet_thresh']:
                A list of class labels per image (one for each task).
                (len(batch_in_label) = batch_size)
            If loss_ind = "contrastive":
                One class label per image. The first half of the list is for
                the left and the second half for the right network input.
                (len(batch_in_label) = 2 x batch_size)
        :batch_in_names (*list (of lists)*)\::
            A list of class labels (names) according to the images in the
            current batch.
            If loss_ind = "triplet_loss":
                One class label per image. (len(batch_in_label) = batch_size)
            If loss_ind in ['soft_triplet', 'triplet_thresh']:
                A list of class labels per image (one for each task).
                (len(batch_in_label) = batch_size)
            If loss_ind = "contrastive":
                Contains the indicator variables instead of the names. If the
                labels of a pair are the same =1 and =0 else.
                (len(batch_in_label) = batch_size)
    """
    batch_in_img    = []
    batch_in_label  = []
    batch_in_names  = []
    num_sample      = 0
    
    if evaluation:
        for image_name in data_dict.keys():
            skip_image=False
            label_names   = []
            label_indexes = []
            for label_key in data_dict[image_name].keys():
                """NEU"""
                if data_dict[image_name][label_key][0] == 'nan':
                    skip_image=True
                    break
                label_names.append(data_dict[image_name][label_key][0])
                label_indexes.append(np.where(np.asarray(label2class_dict[label_key]) ==\
                        data_dict[image_name][label_key][0])[0][0])
            if skip_image:
                continue
            """NEU"""

            image_data = tf.gfile.GFile(image_name, 'rb').read()
            prepro_img = sess.run(in_img_tensor,
                                  feed_dict = {jpeg_data_tensor: image_data})
            batch_in_img.append(prepro_img)
            batch_in_names.append(label_names)
            batch_in_label.append(label_indexes)
            used_images.append(image_name)
            num_sample = num_sample + 1
    
    elif batch_size > 0:
        label2classlist  = {}
        for image_key in data_dict.keys():
            for label_key in data_dict[image_key].keys():
                if label_key not in label2classlist.keys():
                    label2classlist[label_key] = [data_dict[image_key][label_key]]
                else:
                    label2classlist[label_key].append(data_dict[image_key][label_key])
        for label_key in label2classlist.keys():
            label2classlist[label_key] = np.unique(label2classlist[label_key])
        while num_sample < batch_size:   
            image_index = random.randrange(len(list(data_dict.keys())))
            image_name  = list(data_dict.keys())[image_index]
            if image_name not in used_images:
                image_data = tf.gfile.GFile(image_name, 'rb').read()
                prepro_img = sess.run(in_img_tensor,
                                      feed_dict = {jpeg_data_tensor: image_data})
                label_names   = []
                label_indexes = []   
                for label_key in data_dict[image_name].keys():
                    label_names.append(data_dict[image_name][label_key][0])
                    """NEU"""
                    if label_names[-1] == 'nan':
                        label_indexes.append(-1)
                    else:
                        label_indexes.append(np.where(np.asarray(label2class_dict[label_key]) ==\
                                data_dict[image_name][label_key][0])[0][0])
                    """NEU"""
                batch_in_img.append(prepro_img)
                batch_in_names.append(label_names)
                batch_in_label.append(label_indexes)
                used_images.append(image_name)
                num_sample = num_sample + 1
                    
    # take all images in the provided data_dict
    # implicitly used_images has to be empty (cf. run_training())
    elif batch_size < 0:
        for image_name in data_dict.keys():
            label_names   = []
            label_indexes = []
            for label_key in data_dict[image_name].keys():
                label_names.append(data_dict[image_name][label_key][0])
                """NEU"""
                if label_names[-1] == 'nan':
                    label_indexes.append(-1)
                else:
                    label_indexes.append(np.where(np.asarray(label2class_dict[label_key]) ==\
                            data_dict[image_name][label_key][0])[0][0])
                """NEU"""

            image_data = tf.gfile.GFile(image_name, 'rb').read()
            prepro_img = sess.run(in_img_tensor,
                                  feed_dict = {jpeg_data_tensor: image_data})
            batch_in_img.append(prepro_img)
            batch_in_names.append(label_names)
            batch_in_label.append(label_indexes)
            used_images.append(image_name)
            num_sample = num_sample + 1
                
    batch_in_label = np.squeeze(batch_in_label)
    
    return (used_images, batch_in_img, batch_in_label, batch_in_names)


def subdivide_data_dict(data_dict, num_cv_iter, loss_ind):
    r"""Splits the data for cross validation.
    
    The amount of all images per class label are equally split into num_cv_iter
    parts (if the total number of images can be exactly divided by num_cv_iter,
    the last image set is may a little larger or smaller). The images for the
    individual sets are drawn randomly so that each data collection contributes
    roughly equally to each set.
    
    :Arguments\::
        :data_dict (*dictionary*)\::
            If loss_ind in ["triplet_loss", "contrastive"]:
                A dictionary containing the assignments of images to the
                classes. It's data_dict[class_label][img1, ..., imgN], where
                "img" is the absolute path to the image including the image
                name.
            If loss_ind in ['soft_triplet', 'triplet_thresh',
            'soft_contrastive', 'contrastive_thres']:
                A dictionary containing the assignments of images to the
                class labels. It's data_dict[image][task][class], where "image"
                is the absolute path to the image including the image name.
        :num_cv_iter (*int*)\::
            The number of cross validation interations that shall be realized
            on the data in data_dict.
        :loss_ind (*string*)\::
            The loss function that shall be utilized to estimate the network
            performance during training. Possible options:
                'contrastive'       (cf. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964)
                'soft_contrastive' (own development)
                'contrastive_thres' (own development)
                'magnet_loss'       (cf. https://arxiv.org/pdf/1511.05939.pdf)
                'triplet_loss'      (cf. https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
                'soft_triplet'  (own development; ~triplet_multi)
                'triplet_thres'     (own development)   
            
    :Returns\::
        :sd_data_dict (*dictionary*)\::
            It's a dictionary containing num_cv_iter keys and as values a
            data_dict that is a num_cv_iter-th part of the input data_dict.
            The the structure of data_dict is the same as for the input
            data_dict.
    """
    sd_data_dict = {}
    if loss_ind in ['triplet_loss', 'contrastive']:
        for label_key in data_dict.keys():
            all_im_index    = []
            num_images_per_cv = round(len(data_dict[label_key])/num_cv_iter)
            for cv_iter in range(num_cv_iter-1):
                temp_image_list = []
                temp_dict = {}
                for cur in range(num_images_per_cv):
                    image_found = False
                    while not image_found:
                        image_index = random.randrange(len(data_dict[label_key]))
                        if image_index not in all_im_index:
                            temp_image_list.append(data_dict[label_key][image_index])
                            all_im_index.append(image_index)
                            image_found = True
                temp_dict[label_key] = temp_image_list
                if cv_iter not in sd_data_dict.keys():
                    sd_data_dict[cv_iter] = temp_dict
                else:
                    sd_data_dict[cv_iter][label_key] = temp_image_list
            temp_image_list = []
            temp_dict = {}
            # gets all remaining images if the total amount of data
            # can't be divided exactly by the number of cv iterations
            for image_index, image_name in enumerate(data_dict[label_key]):
                if image_index not in all_im_index:
                    temp_image_list.append(image_name)
            temp_dict[label_key] = temp_image_list
            if cv_iter+1 not in sd_data_dict.keys():
                sd_data_dict[cv_iter+1] = temp_dict
            else:
                sd_data_dict[cv_iter+1][label_key] = temp_image_list
    elif loss_ind in ['soft_triplet', 'triplet_thresh',
                      'soft_contrastive', 'contrastive_thres',
                      'soft_contrastive_incomp_loss','soft_triplet_incomp_loss']:
        num_im_per_iter = int(len(list(data_dict.keys()))/num_cv_iter)
        all_im_index    = []  
        for cv_iter in range(num_cv_iter-1):
            temp_image_dict = {}
            for cur in range(num_im_per_iter):
                image_found = False
                while not image_found:
                    image_index = random.randrange(len(list(data_dict.keys())))
                    if image_index not in all_im_index:
                        image_name = list(data_dict.keys())[image_index]
                        temp_image_dict[image_name] = data_dict[image_name]
                        all_im_index.append(image_index)
                        image_found = True
            sd_data_dict[cv_iter] = temp_image_dict
        temp_image_dict = {}
        # gets all remaining images if the total amount of data
        # can't be divided exactly by the number of cv iterations
        for image_index, image_name in enumerate(list(data_dict.keys())):
            if image_index not in all_im_index:
                temp_image_dict[image_name] = data_dict[image_name]
        sd_data_dict[cv_iter+1] = temp_image_dict
        
    return sd_data_dict


def _pairwise_distances(embeddings, squared=False):
    r"""Compute the 2D matrix of distances between all the embeddings.
    
    Source: https://omoindrot.github.io/triplet-loss
    
    :Arguments\::
        :embeddings (*tensor*)\::
            A tensor of shape = (batch_size, num_features) that contain the
            extracted features in case of fed data.
        :squared (*bool*)\::
            If True, output is the pairwise squared euclidean distance matrix.
            If False, output is the pairwise euclidean distance matrix.

    :Returns\::
        :pairwise_distances (*tensor*):
            A tensor of shape = (batch_size, batch_size) containing the
            eucildean distances between all pairs of feature vectors
            (embeddings) in the batch.
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_incomplete_similarity(input_names_l, input_names_r):
    r"""Calculates the similarity degree, considering that some labels may be unknown.
    
    :Arguments\::
        :input_names_l (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the string names of the labels. Belongs to
            feature_tensor_l.
        :input_names_r (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the string names of the labels. Belongs to
            feature_tensor_r.
            
    :Returns\::
        :label_similarity_pos (*tensor*)\::
            A rank-1 tf.float Tensor containing the positive label similarity.
        :label_similarity_neg (*tensor*)\::
            A rank-1 tf.float Tensor containing the negative label similarity.
    """
    # Compute label similarity
    label_equality_p = tf.cast(tf.math.equal(input_names_l, input_names_r), tf.float32)
    label_equality_n = tf.cast(tf.math.not_equal(input_names_l, input_names_r), tf.float32)
    label_nan_l      = tf.cast(tf.math.not_equal(input_names_l, 'nan'), tf.float32)
    label_nan_r      = tf.cast(tf.math.not_equal(input_names_r, 'nan'), tf.float32)
    
    # Normalize by number of tasks
    label_similarity_pos = tf.math.reduce_sum(label_equality_p * label_nan_l * label_nan_r, axis=1) / tf.cast(tf.shape(input_names_l)[1], tf.float32)
    label_similarity_neg = tf.math.reduce_sum(label_equality_n * label_nan_l * label_nan_r, axis=1) / tf.cast(tf.shape(input_names_l)[1], tf.float32)


    return label_similarity_pos, label_similarity_neg


def _get_similarity_degree(input_labels_l, input_labels_r,
                           label_weight_tensor):
    r"""Estimates the weightes multi-label-based similarity.
    
    :Arguments\::
        :input_labels_l (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels. Belongs to
            feature_tensor_l.
        :input_labels_r (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels. Belongs to
            feature_tensor_r.
        :label_weight_tensor (*tensor*)\::
            Contains the label weights of the batch images'labels. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the float weights (sum(weights) = 1) assigned to the
            labels. Indicate the importance of the label for similarity.
            
    :Returns\::
        :similarity_tensor (*tensor*)\::
            A tf.float tensor with shape = [batch_size, 1] containing
            indicators to which degree the class labels of the left and right
            input are the same. The values are in the range of [0; 1], where
            0 means no similarity and 1 means 100% similar.
    """
    # shape = [batch_size, num_rel_labels]
    label_equality = tf.math.equal(input_labels_l, input_labels_r)
    label_equality = tf.cast(label_equality, tf.float32)
    
    # shape = [batch_size, num_rel_labels]
    similarity_impact = tf.multiply(label_weight_tensor,
                                    label_equality)   
    
    # shape = [batch_size]
    similarity_tensor = tf.math.reduce_sum(similarity_impact, axis=1)
    
    return similarity_tensor
    

def _get_triplet_mask_label_similarity(MTL_labels, label_weight_tensor):
    r"""Estimates valid triplets.
    
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - S(x1,x2,L) > S(x,x3,L)
        
    :Arguments\::
        :MTL_labels (*tensor*)\::
            A tf.int32 `Tensor` with shape = [batch_size, num_tasks] containing
            the class labels for all images in the batch (multiple labels per
            image).
    
    :Returns\::
        :mask (*tensor*)\::
            A 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is
            valid. A triplet consists of an anchor (a), a positive sample with
            the same class label as the anchor (p) and a negative sample with
            another class label than the anchor (n).
    """    
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(MTL_labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j,
                                                     i_not_equal_k),
                                      j_not_equal_k)
    
    # check if S(x1,x2,L) > S(x,x3,L)
    margin = _get_margin_from_label_similarity(MTL_labels, label_weight_tensor)
    
    x = tf.fill(tf.shape(margin), False)
    y = tf.fill(tf.shape(margin), True)
    mask_similarity = tf.where(margin <= 0, x, y)
    
    mask = tf.logical_and(distinct_indices, mask_similarity)
    
    return mask


def _get_triplet_mask_label_similarity_incomp(MTL_labels): 
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(MTL_labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j,
                                                     i_not_equal_k),
                                      j_not_equal_k)
    
    # check if S(x1,x2,L) > S(x,x3,L)
    margin = _get_margin_from_label_similarity_incomp(MTL_labels)
    
    x = tf.fill(tf.shape(margin), False)
    y = tf.fill(tf.shape(margin), True)
    mask_similarity = tf.where(margin <= 0, x, y)
    
    mask = tf.logical_and(distinct_indices, mask_similarity)
    
    return mask


def _get_margin_from_label_similarity_incomp(MTL_labels):
    
    # shape = (batch_size, 1, num_rel_labels)
    label_acc_x1x3 = tf.expand_dims(MTL_labels, 1)  
    
    # shape = (batch_size, batch_size, num_rel_labels)     
    label_equality_p = tf.math.equal(MTL_labels, label_acc_x1x3)
    label_equality_p = tf.cast(label_equality_p, tf.float32)
    
    label_equality_n = tf.math.not_equal(MTL_labels, label_acc_x1x3)
    label_equality_n = tf.cast(label_equality_n, tf.float32)
    
    # shape = (batch_size, num_rel_labels)
    label_nan = tf.cast(tf.math.not_equal(MTL_labels, 'nan'), tf.float32)
    
    # shape = (batch_size, batch_size)     
    label_similarity_p = tf.math.reduce_sum(label_equality_p * label_nan, axis=2) / tf.cast(tf.shape(MTL_labels)[1], tf.float32)
    label_similarity_n = tf.math.reduce_sum(label_equality_n * label_nan, axis=2) / tf.cast(tf.shape(MTL_labels)[1], tf.float32)

    # shape = (batch_size, batch_size, 1)
    label_similarity_p = tf.expand_dims(label_similarity_p, 2)
    
    # shape = (batch_size, 1, batch_size)
    label_similarity_n = tf.expand_dims(label_similarity_n, 1)
    
    # shape = (batch_size, batch_size, batch_size) 
    margin = tf.minimum(label_similarity_p, label_similarity_n)
    
    return margin


def _get_margin_from_label_similarity(MTL_labels, label_weight_tensor):
    r"""Estimates the margins for all triplets in the batch.
    
    Note:
        The margins are computed for ALL triplets that can be built in the
        batch. This includes (at this point) also invalid triplets.
    
    :Arguments\::
        :MTL_labels (*tensor*)\::
            A tf.int32 `Tensor` with shape = [batch_size, num_tasks] containing
            the class labels for all images in the batch (multiple labels per
            image).
        :label_weight_tensor (*tensor*)\::
            A tf.float32 tensor with shape = [num_tasks] containing weights
            for the relevance of the tasks' labels for the similarity
            estimation.
    
    :Returns\::
        :margin (*tensor*)\::
            A tf.float32 tensor with shape = (batch_size, batch_size, batch_size)
            ccontainng the margins for all possible triplets in the batch.
    """
    # shape = (batch_size, 1, num_rel_labels)
    label_acc_x1x3 = tf.expand_dims(MTL_labels, 1)  
    
    # shape = (batch_size, batch_size, num_rel_labels)     
    label_equality = tf.math.equal(MTL_labels, label_acc_x1x3)
    label_equality = tf.cast(label_equality, tf.float32)
    
    # shape = (num_rel_labels)
    similarity_impact = tf.multiply(label_weight_tensor,
                                    label_equality)      
    
    # shape = (batch_size, batch_size)
    pairwise_similarity = tf.math.reduce_sum(similarity_impact, axis=2)
    
    # shape = (batch_size, batch_size, 1)
    x1x2_positive_sim = tf.expand_dims(pairwise_similarity, 2)
    
    # shape = (batch_size, 1, batch_size)
    x1x3_negative_sim = tf.expand_dims(pairwise_similarity, 1)
    
    # shape = (batch_size, batch_size, batch_size) 
    margin = x1x2_positive_sim - x1x3_negative_sim
    
    return margin


def _get_mask_hard_triplet_loss(MTL_labels, label_weight_tensor,
                                similarity_thresh):
    r"""Estimates threshold-based valid triplets.
    
    Invalid triplets are
    where i, j, k not distinct
    where S(x_a, x_p) < similarity_thresh (has to be larger to be a positive sample)
    where S(x_a, x_n) >=similarity_thresh (has to be smaller to be a negative sample)
    
    :Arguments\::
        :MTL_labels (*tensor*)\::
            A tf.int32 `Tensor` with shape = [batch_size, num_tasks] containing
            the class labels for all images in the batch (multiple labels per
            image).
        :label_weight_tensor (*tensor*)\::
            A tf.float32 tensor with shape = [num_tasks] containing weights
            for the relevance of the tasks' labels for the similarity
            estimation.
        :similarity_thresh (*float*)\::
            Has to be provided, if loss_ind is "triplet_thresh".
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
    
    :Returns\::
        :mask (*tensor*)\::
            A 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is
            valid. A triplet consists of an anchor (a), a positive sample with
            the same class label as the anchor (p) and a negative sample with
            another class label than the anchor (n).
    """
    # Check that i, j and k are distinct
    # valid triplets regarding the indices
    indices_equal = tf.cast(tf.eye(tf.shape(MTL_labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
 
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j,
                                                     i_not_equal_k),
                                      j_not_equal_k)
                
    # valid triplets regarding the label similarity
    # and the distinct indices of (i, j), (i, k)
    # -> if the similarity_thresh < 0.5, (j, k) can have but must not have the
    #    same indices
    # -> different indices are only guaranteed, if the label similarity of
    #    (j, k) is smaller than similarity_thresh
    mask_pos     = _get_triplet_mask_positive(MTL_labels,
                                              label_weight_tensor,
                                              similarity_thresh)
    mask_neg     = _get_triplet_mask_negative(MTL_labels,
                                              label_weight_tensor,
                                              similarity_thresh)
    
    mask_pos_aux = tf.expand_dims(tf.cast(mask_pos, tf.bool), 1)
    mask_neg_aux = tf.expand_dims(tf.cast(mask_neg, tf.bool), 2)
    
    # implicitly (i, j, k) have distinct indices
    # -> i, j (anchor, positive) have distinct indices (and similar labels)
    # -> i, k (anchor, negative) have dissimillar labels and thus distinct
    #                            indices
    # -> implicitly, (j, k) have dissimilar labels!?
    #    JUST THE CASE FOR similarity_thresh >= 0.5!!!
    # -> thus, (j, k) can not have the same indices
    mask = tf.logical_and(mask_pos_aux, mask_neg_aux)
    mask = tf.logical_and(mask, distinct_indices)
    
    return mask


def _get_triplet_mask_positive(MTL_labels, label_weight_tensor,
                               similarity_thresh):
    r"""Estimates all positives for all anchors.
    
    A sample is positive for an anchor sample, if the weighted sum of label
    similarities is larger than a threshold and the samples are distinct.
    
    :Arguments\::
        :MTL_labels (*tensor*)\::
            A tf.int32 `Tensor` with shape = [batch_size, num_tasks] containing
            the class labels for all images in the batch (multiple labels per
            image).
        :label_weight_tensor (*tensor*)\::
            A tf.float32 tensor with shape = [num_tasks] containing weights
            for the relevance of the tasks' labels for the similarity
            estimation.
        :similarity_thresh (*float*)\::
            Has to be provided, if loss_ind is "triplet_thresh".
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
    
    :Returns\::
        :mask (*tensor*)\::
            A tf.bool `Tensor` with shape = [batch_size, batch_size]
            containing a 2D mask where mask[a, p] is True if a (anchor) and p
            (positive sample) are distinct and have same label.
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(MTL_labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    
    # shape = (batch_size, 1, num_rel_labels)
    label_acc_x1x3 = tf.expand_dims(MTL_labels, 1)  
    
    # shape = (batch_size, batch_size, num_rel_labels)     
    label_equality = tf.math.equal(MTL_labels, label_acc_x1x3)
    label_equality = tf.cast(label_equality, tf.float32)
    
    # shape = (num_rel_labels)
    similarity_impact = tf.multiply(label_weight_tensor,
                                    label_equality)      
    
    # shape = (batch_size, batch_size)
    pairwise_similarity = tf.math.reduce_sum(similarity_impact, axis=2)
    threshold_pos       = tf.math.greater_equal(pairwise_similarity,
                                                similarity_thresh)
    
    # shape = (batch_size, batch_size)
    bool_mask = tf.logical_and(indices_not_equal, threshold_pos)
    mask      = tf.to_float(bool_mask)
    
    return mask


def _get_triplet_mask_negative(MTL_labels, label_weight_tensor,
                               similarity_thresh):
    r"""Estimates all negatives for all anchors.
    
    A sample is negative for an anchor sample, if the weighted sum of label
    similarities is smaller than a threshold.
    
    :Arguments\::
        :MTL_labels (*tensor*)\::
            A tf.int32 `Tensor` with shape = [batch_size, num_tasks] containing
            the class labels for all images in the batch (multiple labels per
            image).
        :label_weight_tensor (*tensor*)\::
            A tf.float32 tensor with shape = [num_tasks] containing weights
            for the relevance of the tasks' labels for the similarity
            estimation.
        :similarity_thresh (*float*)\::
            Has to be provided, if loss_ind is "triplet_thresh".
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
    
    :Returns\::
        :mask (*tensor*)\::
            A tf.bool `Tensor` with shape = [batch_size, batch_size]
            containing a 2D mask where mask[a, n] is True if a (anchor) and n
            (negative sample) have distinct labels (implicitly the case).
    """
    # shape = (batch_size, 1, num_rel_labels)
    label_acc_x1x3 = tf.expand_dims(MTL_labels, 1)  
    
    # shape = (batch_size, batch_size, num_rel_labels)     
    label_equality = tf.math.equal(MTL_labels, label_acc_x1x3)
    label_equality = tf.cast(label_equality, tf.float32)
    
    # shape = (num_rel_labels)
    similarity_impact = tf.multiply(label_weight_tensor,
                                    label_equality)      
    
    # shape = (batch_size, batch_size)
    pairwise_similarity = tf.math.reduce_sum(similarity_impact, axis=2)
    threshold_pos       = tf.math.greater(pairwise_similarity,
                                          similarity_thresh)
    adapted_neg         = tf.logical_not(threshold_pos)
    mask                = tf.to_float(adapted_neg)
    
    return mask


def soft_contrastive_incomp_loss(feature_tensor_l, feature_tensor_r,
                                 input_names_l, input_names_r):
    r"""Estimates the constrastive loss for multi-label-based similarity.
    
    :Arguments\::
        :feature_tensor_l (*tensor*)\::
            A tf.float tensor with shape = [batch_size, num_features]
            containing the features of the "left" network input.
        :feature_tensor_r (*tensor*)\::
            A tf.float tensor with shape = [batch_size, num_features]
            containing the features of the "right" network input.
        :input_names_l (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the string names of the labels. Belongs to
            feature_tensor_l.
        :input_names_r (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the string names of the labels. Belongs to
            feature_tensor_r.
            
    :Returns\::
        :loss (*tensor*)\::
            A scalar tf.float Tensor containing the contrastive loss.
        :bool_reduce_mean (*bool*)\::
            Whether to reduece the losses of the batch to one mean loss or not.
    """
    with tf.name_scope("soft_contrastive_incomp_loss"):
        
        # Epsilon for Norm to avoid NaN in gradients
        eps = tf.constant(1e-15, dtype=tf.float32)
        
        # Distance d
        distance = tf.squeeze(tf.sqrt(tf.reduce_sum(
                                tf.pow(feature_tensor_l - feature_tensor_r + eps, 2),
                                1,
                                keepdims=True)+eps))
        
        # Label similarity Yp, Yn
        Yp, Yn = _get_incomplete_similarity(input_names_l, input_names_r)
        
        # positive and negative margins Mp, Mn
        l_is_nan = tf.cast(tf.math.equal(input_names_l, 'nan'), tf.float32)
        r_is_nan = tf.cast(tf.math.equal(input_names_r, 'nan'), tf.float32)
        Mp = tf.reduce_sum(tf.minimum(1., l_is_nan+r_is_nan), axis=1) \
             / tf.cast(tf.shape(input_names_l)[1], tf.float32)
        Mn = 2.-Mp
        
        loss = Yp * tf.maximum(0., distance - Mp) + Yn * tf.maximum(0., Mn - distance)
        
       
        
        bool_reduce_mean    = tf.placeholder(tf.bool)
        loss = tf.cond(bool_reduce_mean,
                           lambda: tf.reduce_mean(loss),
                           lambda: loss)
        
        return loss, bool_reduce_mean
    

def hard_contrastive_loss(feature_tensor_l, feature_tensor_r, input_labels_l,
                          input_labels_r, label_weight_tensor, margin,
                          similarity_thresh):
    r"""Estimates the constrastive loss for multi-label-based similarity.
    
    :Arguments\::
        :feature_tensor_l (*tensor*)\::
            A tf.float tensor with shape = [batch_size, num_features]
            containing the features of the "left" network input.
        :feature_tensor_r (*tensor*)\::
            A tf.float tensor with shape = [batch_size, num_features]
            containing the features of the "right" network input.
        :input_labels_l (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels. Belongs to
            feature_tensor_l.
        :input_labels_r (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels. Belongs to
            feature_tensor_r.
        :label_weight_tensor (*tensor*)\::
            Contains the label weights of the batch images'labels. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the float weights (sum(weights) = 1) assigned to the
            labels. Indicate the importance of the label for similarity.
        :margin (*float*)\::
            A scalar defining the minimum distance that should be between non-
            matching pairs of features in feature space.
        :similarity_thresh (*float*)\::
            Has to be provided, if loss_ind is "triplet_thresh".
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
            
    :Returns\::
        :loss (*tensor*)\::
            A scalar tf.float Tensor containing the contrastive loss.
        :bool_reduce_mean (*bool*)\::
            Whether to reduece the losses of the batch to one mean loss or not.
    """
    with tf.name_scope("hard_contrastive_loss"):
        
        # Epsilon for Norm to avoid NaN in gradients
        eps = tf.constant(1e-15, dtype=tf.float32)
        
        distance = tf.squeeze(tf.sqrt(tf.reduce_sum(
                                tf.pow(feature_tensor_l - feature_tensor_r + eps, 2),
                                1,
                                keepdims=True)+eps))
        similarity_degree = _get_similarity_degree(input_labels_l,
                                                   input_labels_r,
                                                   label_weight_tensor)
        similarity_ind_match = tf.math.greater_equal(similarity_degree,
                                                     similarity_thresh)
        similarity_tensor = tf.to_float(similarity_ind_match)
        
        # keep the similar label (1) close to each other
        similarity = similarity_tensor * tf.square(distance+eps) 
        
        # give penalty to dissimilar label if the distance is smaller than margin                                          
        dissimilarity = (1 - similarity_tensor) * tf.square(
                        tf.maximum((margin - distance), 0)+eps)  
        
        bool_reduce_mean    = tf.placeholder(tf.bool)
        loss = tf.cond(bool_reduce_mean,
                       lambda: tf.reduce_mean(dissimilarity + similarity) / 2,
                       lambda: (dissimilarity + similarity) / 2)
        
        return loss, bool_reduce_mean
    

def soft_contrastive_loss(feature_tensor_l, feature_tensor_r, input_labels_l,
                          input_labels_r, label_weight_tensor, margin):
    r"""Estimates the constrastive loss for multi-label-based similarity.
    
    :Arguments\::
        :feature_tensor_l (*tensor*)\::
            A tf.float tensor with shape = [batch_size, num_features]
            containing the features of the "left" network input.
        :feature_tensor_r (*tensor*)\::
            A tf.float tensor with shape = [batch_size, num_features]
            containing the features of the "right" network input.
        :input_labels_l (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels. Belongs to
            feature_tensor_l.
        :input_labels_r (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels. Belongs to
            feature_tensor_r.
        :label_weight_tensor (*tensor*)\::
            Contains the label weights of the batch images'labels. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the float weights (sum(weights) = 1) assigned to the
            labels. Indicate the importance of the label for similarity.
        :margin (*float*)\::
            A scalar defining the minimum distance that should be between non-
            matching pairs of features in feature space.
            
    :Returns\::
        :loss (*tensor*)\::
            A scalar tf.float Tensor containing the contrastive loss.
        :bool_reduce_mean (*bool*)\::
            Whether to reduece the losses of the batch to one mean loss or not.
    """
    with tf.name_scope("soft_contrastive_loss"):
        
        # Epsilon for Norm to avoid NaN in gradients
        eps = tf.constant(1e-15, dtype=tf.float32)
        
        distance = tf.squeeze(tf.sqrt(tf.reduce_sum(
                                tf.pow(feature_tensor_l - feature_tensor_r + eps, 2),
                                1,
                                keepdims=True)+eps))
        similarity_tensor = _get_similarity_degree(input_labels_l,
                                                   input_labels_r,
                                                   label_weight_tensor)
        # keep the similar label (1) close to each other
        similarity = similarity_tensor * tf.square(distance+eps) 
        # give penalty to dissimilar label if the distance is bigger than margin                                          
        dissimilarity = (1 - similarity_tensor) * tf.square(
                        tf.maximum((margin - distance), 0)+eps) 
        
        bool_reduce_mean    = tf.placeholder(tf.bool)
        loss = tf.cond(bool_reduce_mean,
                           lambda: tf.reduce_mean(dissimilarity + similarity) / 2,
                           lambda: (dissimilarity + similarity) / 2)
        
        return loss, bool_reduce_mean
    

def batch_all_hard_triplet_loss(input_labels, embeddings, margin,
                                   label_weight_tensor, similarity_thresh):
    r"""Triplet loss for label-based thresholded similarity.
    
    :Arguments\::
        :input_labels (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels.
        :embeddings (*tensor*)\::
            Contains the extracted features of the batch images. Has the shape
            (batch_size, num_features), where num_labels = #(nodes of last layer).
        :margin (*float*)\::
            margin for triplet loss
        :label_weight_tensor (*tensor*)\::
            Contains the label weights of the batch images'labels. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the float weights (sum(weights) = 1) assigned to the
            labels. Indicate the importance of the label for similarity.
        :similarity_thresh (*float*)\::
            Has to be provided, if loss_ind is "triplet_thresh".
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
    
    :Returns\::
        :triplet_loss (*tensor*)\::
            It's a scalar float tensor containing the batch loss.   
        :bool_reduce_sum (*bool*)\::
            Whether to reduece the losses of the batch to one mean loss or not.
            The mean is only estimated among the losses of valid triplets.
    """
    MTL_labels = input_labels
    
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=False)

    x1x2_positive_dist = tf.expand_dims(pairwise_dist, 2)
    x1x3_negative_dist = tf.expand_dims(pairwise_dist, 1)
    
    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of x1=i, x2=j, x3=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = x1x2_positive_dist - x1x3_negative_dist + margin

    # Put to zero the invalid triplets
    # where i, j, k not distinct
    # where S(x_a, x_p) < similarity_thresh (has to be larger to be a positive sample)
    # where S(x_a, x_n) >=similarity_thresh (has to be smaller to be a negative sample)
    mask         = _get_mask_hard_triplet_loss(MTL_labels,
                                               label_weight_tensor,
                                               similarity_thresh)   
    mask         = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    # implicitly removes easy triplets by averaging over all hard and
    # semi-hard triplets
    # (already correct distance of triplets' features in feature space)
    valid_triplets             = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets      = tf.reduce_sum(valid_triplets) # valid triplets in the sense that they are valid and that they are no easy triplets
#    num_valid_triplets         = tf.reduce_sum(mask) # valid triplets only in the sense that they are valid
#    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    bool_reduce_sum    = tf.placeholder(tf.bool)
    triplet_loss = tf.cond(bool_reduce_sum,
                           lambda: tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16),
                           lambda: triplet_loss)
    
    return triplet_loss, bool_reduce_sum


def all_label_similarity_triplet_loss(input_labels, embeddings,
                                      label_weight_tensor):
    r"""Triplet loss for label-based soft assigmnets of similarity.
    
    The standard triplet loss uses a positive and a negative sample per anchor
    in a triplet.
    This triplet loss uses softer decision criterions regarding "positive and
    negative samples". Sample x1 (equivalent to "anchor") has to be more similar
    to x2 (roughly "positive") than to x3 (roughly "negative"). The loss is
    computed via:
        L(x1,x2,x3,L) = max[0, M(x1,x2,x3,L) + d(x1,x2)) - d(x1,x3)]    
    
        M(x1,x2,x3,L) = S(x1,x2,L) - S(x,x3,L)
        
        S(x1,x2,L)    = sum_L(label_weight * Kron_delta(label(x1), label(x2)))
        
        L(x)          = [label_1(x), ..., label_N(x)]
        
        d(x1,x2)      = euclidean distance between the feature vectors x1, x2
        
    Invalid triplets have the property that:
        S(x1,x2,L) <= S(x,x3,L)        
    
    :Arguments\::
        :input_labels (*tensor*)\::
            Contains the labels of the batch images. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the int indexes of the labels.
        :embeddings (*tensor*)\::
            Contains the extracted features of the batch images. Has the shape
            (batch_size, num_features), where num_labels = #(nodes of last layer).
        :label_weight_tensor (*tensor*)\::
            Contains the label weights of the batch images'labels. Has the shape
            (batch_size, num_lables), where num_labels = len(relevant_variables).
            The entries are the float weights (sum(weights) = 1) assigned to the
            labels. Indicate the importance of the label for similarity.
    
    :Returns\::
        :triplet_loss (*tensor*)\::
            It's a scalar float tensor containing the batch loss.
        :fraction_positive_triplets (*tensor*)\::
            A scalar tf.float tensor containing the amount of hard and semi-
            hard triplets (positive loss) among all valid triplets.
        :bool_reduce_sum (*bool*)\::
            Whether to reduece the losses of the batch to one mean loss or not.
            The mean is only estimated among the losses of valid triplets.
    """
    MTL_labels = input_labels
    
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=False)

    x1x2_positive_dist = tf.expand_dims(pairwise_dist, 2)
    x1x3_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # shape(margin) = (batch_size, batch_size, batch_size)
    margin = _get_margin_from_label_similarity(MTL_labels, label_weight_tensor)
    
    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of x1=i, x2=j, x3=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = x1x2_positive_dist - x1x3_negative_dist + margin

    # Put to zero the invalid triplets
    # where i, j, k not distinct
    # where S(x1,x2) <= S(x1,x3)
    mask         = _get_triplet_mask_label_similarity(MTL_labels,
                                                      label_weight_tensor)
    mask         = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    # implicitly removes easy triplets by averaging over all hard and
    # semi-hard triplets
    # (already correct distance of triplets' features in feature space)
    valid_triplets             = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets      = tf.reduce_sum(valid_triplets) # valid triplets in the sense that they are valid and that they are no easy triplets
    num_valid_triplets         = tf.reduce_sum(mask) # valid triplets only in the sense that they are valid
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    bool_reduce_sum    = tf.placeholder(tf.bool)
    triplet_loss = tf.cond(bool_reduce_sum,
                           lambda: tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16),
                           lambda: triplet_loss)

    return triplet_loss, fraction_positive_triplets, bool_reduce_sum


def all_label_similarity_triplet_incomp_loss(input_labels, embeddings):
    
    MTL_labels = input_labels
    
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=False)

    x1x2_positive_dist = tf.expand_dims(pairwise_dist, 2)
    x1x3_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # shape(margin) = (batch_size, batch_size, batch_size)
    margin = _get_margin_from_label_similarity_incomp(MTL_labels)
    
    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of x1=i, x2=j, x3=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = x1x2_positive_dist - x1x3_negative_dist + margin

    # Put to zero the invalid triplets
    # where i, j, k not distinct
    # where S(x1,x2) <= S(x1,x3)
    mask         = _get_triplet_mask_label_similarity_incomp(MTL_labels)
    mask         = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    # implicitly removes easy triplets by averaging over all hard and
    # semi-hard triplets
    # (already correct distance of triplets' features in feature space)
    valid_triplets             = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets      = tf.reduce_sum(valid_triplets) # valid triplets in the sense that they are valid and that they are no easy triplets
    num_valid_triplets         = tf.reduce_sum(mask) # valid triplets only in the sense that they are valid
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    bool_reduce_sum    = tf.placeholder(tf.bool)
    triplet_loss = tf.cond(bool_reduce_sum,
                           lambda: tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16),
                           lambda: triplet_loss)

    return triplet_loss, fraction_positive_triplets, bool_reduce_sum


def create_module_graph(tfhub_module, module_spec, add_fc, retrain,
                        reuse=False):
    r"""Creates a graph and loads the Hub Module into it.

    :Arguments:
        :tfhub_module (*string*)\::
            This variabel is a string and contains the Module URL to the
            desired networks feature vector. For ResNet-152 V2 is has to be
            'https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/1'.
            Other posibilities for feature vectors can be found at
            'https://tfhub.dev/s?module-type=image-feature-vector'.
        :module_spec (*hub module*)\::
            The hub.ModuleSpec for the image module being used.
        :add_fc (*array of int*)\::
            Has to be set if bool_hub_module is True.
            The number of fully connected layers that shall be trained on top
            of the tensorflow hub Module is equal to the number of entries in
            the array. Each entry is an int specifying the number of nodes in
            the individual fully connected layers. If [1000, 100] is given, two
            fc layers will be added, where the first has 1000 nodes and the
            second has 100 nodes. If no layers should be added, an empty array
            '[]' has to be given.

            Warning:
                The last layer should not be too high-dimensional (>> 20),
                because of the nearest-neighbor queries' dimensionality in
                evaluation.
        :retrain (*int*)\::
            The number of layers (or blocks in case of ResNet) from the
            tensorflow hub Module that shall be retrained.
        :reuse (*bool*)\::
            Whether the weights and biases of the layers shall be reused (in
            case that this function is called at least twice) or not.

    :Returns:
        :in_batch_tensor (*tensor*)\::
            A tensor that has to be fed with data and builds the input for the
            network. It has the shape = (batch_size, height, width, 3).
        :output_feature_tensor (*tensor*)\::
            The last/output tensor of the CNN containing the extracted
            features in case of fed data. It has the
            shape = (1, batch_size, num_features).
        :retrain_vars (*list*)\::
            Contains all names of the layers in the hub module that shall be
            retrained.
    """

    """**************MODULE GRAPH********************"""
    height, width = hub.get_expected_image_size(module_spec)
    with tf.variable_scope("ModuleLayers"):
        in_batch_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name="input_img")
    # with tf.variable_scope("ModuleLayers"):
    #     in_batch_tensor = tf.placeholder(tf.float32, [None, height, width, 3], name="input_img")
    #     module        = hub.Module(module_spec, trainable=True)
    #     output_module = module(in_batch_tensor)

    if retrain > 0 and not reuse:
        if 'resnet' in tfhub_module:
            # feature computation
            module = hub.Module(module_spec, trainable=True)
            output_module = module(in_batch_tensor)
            # get variables to retrain
            help_string_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope='module/resnet')[0].name
            help_string_2 = help_string_1.split('/')[0] + '/' + \
                            help_string_1.split('/')[1] + '/block'

            temp_choice = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope=help_string_2)
            max_block = 0
            max_unit = 0
            residual_block_dict = {}
            num_residual_blocks = 0
            for variable in temp_choice:
                temp_block = int(variable.name.split('block')[1].split('/')[0])
                temp_unit = int(variable.name.split('unit_')[1].split('/')[0])
                if temp_block not in residual_block_dict.keys():
                    residual_block_dict[temp_block] = [temp_unit]
                else:
                    residual_block_dict[temp_block].append(temp_unit)
                if temp_block > max_block:
                    max_block = temp_block
                    num_residual_blocks += max_unit
                    max_unit = 0
                if temp_unit > max_unit:
                    max_unit = temp_unit
            num_residual_blocks += max_unit

            num_added_res_blocks = 0
            retrain_vars = []
            for makro_block in range(max_block, 0, -1):
                for res_block in range(max(residual_block_dict[makro_block]), 0, -1):
                    if num_added_res_blocks < retrain:
                        retrain_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                              scope=help_string_2 \
                                                                    + str(makro_block) \
                                                                    + '/unit_' + str(res_block)))
                        num_added_res_blocks += 1
                    else:
                        break
        else:
            print('only implemented for ResNet by now!')

            sys.exit()
    else:
        retrain_vars = []
        m = hub.Module(module_spec)
        output_module = m(in_batch_tensor)

        # # get scope/names of variables from layers that will be retrained
    # module_vars        = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ModuleLayers')
    # pre_names          = '/'.join(module_vars[0].name.split('/')[:3])
    # module_vars_names  = np.asarray([v.name.split('/')[3] for v in module_vars])[::-1]
    #
    # unique_module_vars_names = []
    # for n in module_vars_names:
    #     if len(unique_module_vars_names) == 0 or (not n == unique_module_vars_names[-1]):
    #         unique_module_vars_names += [n]
    #
    # retrain_vars = []
    # for v in range(retrain):
    #     retrain_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=pre_names+'/'+unique_module_vars_names[v]))

    """**************\MODULE GRAPH********************"""

    """**************CUSTOM GRAPH********************"""
    init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
    with tf.variable_scope('CustomLayers'):
        if len(add_fc) == 1:
            output_feature_tensor = tf.layers.dense(inputs=output_module,
                                                  units=add_fc[-1],
                                                  use_bias=True,
                                                  kernel_initializer = init,
                                                  activation=None,
                                                  name='output_features',
                                                  reuse=reuse)
        elif len(add_fc) > 1:
            for cur_fc in range(len(add_fc)-1):
                if cur_fc == 0:
                    dense_layer = tf.layers.dense(inputs=output_module,
                                                  units=add_fc[cur_fc],
                                                  use_bias=True,
                                                  kernel_initializer = init,
                                                  activation=tf.nn.relu,
                                                  name='fc_layer' + str(cur_fc) +\
                                                       '_' + str(add_fc[cur_fc]),
                                                  reuse=reuse)
                else:
                    dense_layer = tf.layers.dense(inputs=dense_layer,
                                                  units=add_fc[cur_fc],
                                                  use_bias=True,
                                                  kernel_initializer = init,
                                                  activation=tf.nn.relu,
                                                  name='fc_layer' + str(cur_fc) +\
                                                       '_' + str(add_fc[cur_fc]),
                                                  reuse=reuse)
            output_feature_tensor = tf.layers.dense(inputs=dense_layer,
                                                  units=add_fc[-1],
                                                  use_bias=True,
                                                  kernel_initializer = init,
                                                  activation=None,
                                                  name='output_features',
                                                  reuse=reuse)
        else:
            output_feature_tensor = output_module
    """**************\CUSTOM GRAPH********************"""

    return in_batch_tensor, output_feature_tensor, retrain_vars


def train_model(master_file, master_dir, logpath,
           train_batch_size, how_many_training_steps, learning_rate,
           tfhub_module, add_fc, hub_num_retrain, aug_dict, optimizer_ind, loss_ind,
           relevant_variables, similarity_thresh, label_weights, 
           how_often_validation, val_percentage):
    r""" Trains a new model for calculating feature vectors.
    
    :Arguments\::
        :master_file (*string*)\::
            The name of the master file that shall be used.
            The master file has to contain a list of the "collection.txt".
            All "collection.txt" files have to be in the same folder as the master
            file. The "collection.txt" files list samples with relative paths to the images and the
            according class labels. The paths in a "collection.txt" have to
            be "some_rel_path_from_location_of_master_txt/image_name.jpg".
            All samples from all stated collection files will be used for training and,
            possibly, validation.
        :master_dir (*string*)\::
            This variable is a string and contains the absolute path to the
            master file.
        :logpath (*string*):
            The path where all summarized training information and the trained
            network will be stored.
        :train_batch_size (*int*)\::
            This variable defines how many images shall be used for
            the classifier's training for one training step.
            Default: XXX.
        :how_many_training_steps (*int*)\::
            Number of training iterations.
        :learning_rate (*float*)\::
            Specifies the learning rate of the Optimizer.
            Default: XXX.
        :tfhub_module (*string*)\::
            This variable contains the Module URL to the
            desired networks feature vector. For ResNet-152 V2 is has to be
            'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1'.
            Other posibilities for feature vectors can be found at
            'https://tfhub.dev/s?module-type=image-feature-vector'.
            Default: XXX.
        :add_fc (*array of int*)\::
            The number of fully connected layers that shall be trained on top
            of the tensorflow hub Module is equal to the number of entries in
            the array. Each entry is an int specifying the number of nodes in
            the individual fully connected layers. If [1000, 100] is given, two
            fc layers will be added, where the first has 1000 nodes and the
            second has 100 nodes. If no layers should be added, an empty array
            '[]' has to be given.
            Default: XXX.
        :hub_num_retrain (*int*)\::
            The number of layers from the
            tensorflow hub Module that shall be retrained. 
            Default: XXX.
        :aug_dict (*dict*)\::
            A dictionary specifying which types of data augmentations shall 
            be applied during training. A list of available augmentations can be
            found in the documentation of the SILKNOW WP4 Library.
            Default: XXX
        :optimizer_ind (*string*)\::
            The optimizer that shall be used during the training procedure of
            the siamese network. Possible options:
                'Adagrad' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
                'Adam' (cf. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
                'GradientDescent' (cf. https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)
            Default: XXX
        :loss_ind (*string*)\::
            The loss function that shall be utilized to estimate the network
            performance during training. Possible options:
                'contrastive'       (cf. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1640964)
                'soft_contrastive' (own development)
                'contrastive_thres' (own development)
                'triplet_loss'      (cf. https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)
                'soft_triplet'  (own development; ~triplet_multi)
                'triplet_thresh'    (own development) 
            Default: XXX
        :relevant_variables (*list*)\::
            A list containing the collection.txt's header terms of the labels
            to be considered. The terms must not contain blank spaces; use '_'
            instead of blank spaces!
            Example (string in control file): #timespan, #place  
            Example (according list in code): [timespan, place]
            Default: XXX
        :similarity_thresh (*float*)\::
            In case of an equal impact of 5 labels, a threshold of:
                - 1.0 means that 5 labels need to be equal
                - 0.8 means that 4 labels need to be equal
                - 0.6 means that 3 labels need to be equal
                - 0.4 means that 2 labels need to be equal
            for similarity.
            Default: XXX
        :label_weights (*list*)\::
            Weights that express the importance of the individual labels for
            the similarity estimation. Have to be positive numbers. Will be
            normalized so that the sum is 1.
            The list has to be as long as "relevant_variables".
            Default: XXX
        :how_often_validation (*int*)\::
            Number of training iterations between validations.
            Default: XXX
        :val_percentage (*int*)\::
            Percentage of training data that will be used for validation.
            Default: XXX
    
    """
    bool_data_aug = True if len(aug_dict.keys()) > 0 else False
    # 1.1 check whether directories exist. If not, create them
    # =================
    if not os.path.exists(os.path.join(logpath,r"")): os.makedirs(os.path.join(logpath,r""))
    
    # 2. collect data samples
    # =======================
    if loss_ind in ['soft_triplet', 'triplet_thresh',
                      'soft_contrastive', 'contrastive_thres',
                      'soft_contrastive_incomp_loss','soft_triplet_incomp_loss']:
        # 1. key:       image_name
        # 2. key/value: task/variable/label
        # value:        class label
        coll_list = sn_func.master_file_to_collections_list(
                master_dir, master_file)
        coll_dict, data_dict = sn_func.collections_list_MTL_to_image_lists(coll_list,
                                                                  relevant_variables,
                                                                  1,
                                                                  master_dir,
                                                                  False)
        label2class_dict = {}
        for label_key in coll_dict.keys():
            label2class_dict[label_key] = list(coll_dict[label_key].keys())
        
    else:
        print('This loss is not implemented yet or may the key is written\
              incorrectly.')        
        sys.exit()
        
    # 3. Build Graph
    # =======================
    graph = tf.Graph()
    with graph.as_default():        
        # 3.1 integrate pre-processing pipeline
        # =====================================
        # jpeg_data_tensor, in_img_tensor will contain one image, not one batch
        # (scale to input size, potential augmentations, ...) 
        module_spec = hub.load_module_spec(str(tfhub_module))
        # (jpeg_data_tensor,
        #  in_img_tensor) = sn_func.add_jpeg_decoding(input_height=None,
        #                                     input_width=None,
        #                                     input_depth=None,
        #                                     module_spec=module_spec,
        #                                     bool_hub_module=True,
        #                                     bool_data_aug=bool_data_aug,
        #                                     aug_set_dict=aug_dict)
        (jpeg_data_tensor,
         in_img_tensor) = sn_func.add_jpeg_decoding(module_spec=module_spec)
        augmented_image_tensor = sn_func.add_data_augmentation(aug_dict, in_img_tensor)
        
        # 3.2 build siamese network...
        # ============================
        (in_batch_tensor,
         output_feature_tensor,
         retrain_vars) = create_module_graph(tfhub_module,
                                             module_spec,
                                             add_fc,
                                             retrain=hub_num_retrain,
                                             reuse=False)
        # ... with normalized feature vectors
        output_feature_tensor = tf.nn.l2_normalize(output_feature_tensor,
                                                   axis=-1)
        
        if loss_ind == 'soft_triplet':
        # ==================================
            in_label_tensor = tf.placeholder(tf.float16,
                                             [None, len(relevant_variables)],
                                             name="in_label_tensor")
            label_weight_tensor = tf.placeholder(tf.float32,
                                             [len(relevant_variables)],
                                             name="label_weight_tensor")
            (loss, _,
             bool_reduce) = all_label_similarity_triplet_loss(
                            input_labels        = in_label_tensor,
                            embeddings          = output_feature_tensor,
                            label_weight_tensor = label_weight_tensor)
            
        elif loss_ind == 'soft_triplet_incomp_loss':    
        # ==================================
        # TODO: D4.6: check how often bad combinations are drawn (crossed nan-settings -> Yp=Yn=0)
            in_label_tensor = tf.placeholder(tf.string,
                                             [None, len(relevant_variables)],
                                             name="in_name_tensor")
            label_weight_tensor = tf.placeholder(tf.float32,
                                             [len(relevant_variables)],
                                             name="label_weight_tensor")
            (loss, _,
             bool_reduce) = all_label_similarity_triplet_incomp_loss(
                            input_labels        = in_label_tensor,
                            embeddings          = output_feature_tensor)
        
        elif loss_ind == 'triplet_thresh':
        # ==================================
            in_label_tensor = tf.placeholder(tf.float16,
                                             [None, len(relevant_variables)],
                                             name="in_label_tensor")
            label_weight_tensor = tf.placeholder(tf.float32,
                                             [len(relevant_variables)],
                                             name="label_weight_tensor")
            (loss,
             bool_reduce) = batch_all_hard_triplet_loss(
                            input_labels        = in_label_tensor,
                            embeddings          = output_feature_tensor,
                            margin              = 0.5,
                            label_weight_tensor = label_weight_tensor,
                            similarity_thresh   = similarity_thresh)

        elif loss_ind == 'soft_contrastive':
        # =============================         
            in_label_tensor = tf.placeholder(tf.float16,
                                             [None, len(relevant_variables)],
                                             name="in_label_tensor")
            label_weight_tensor = tf.placeholder(tf.float32,
                                             [len(relevant_variables)],
                                             name="label_weight_tensor")
            (output_feature_tensor_l,
             output_feature_tensor_r) = tf.split(value=output_feature_tensor,
                                                 num_or_size_splits=2,
                                                 axis=0)
            (in_label_tensor_l,
             in_label_tensor_r) = tf.split(value=in_label_tensor,
                                           num_or_size_splits=2,
                                           axis=0)
            # Validate margin=2 for paper dataset (was 0.5 for ISPRS paper)
            (loss,
             bool_reduce) = soft_contrastive_loss(
                            feature_tensor_l    = output_feature_tensor_l,
                            feature_tensor_r    = output_feature_tensor_r,
                            input_labels_l      = in_label_tensor_l, 
                            input_labels_r      = in_label_tensor_r, 
                            label_weight_tensor = label_weight_tensor,
                            margin              = 2)
        elif loss_ind == 'contrastive_thres':
        # =============================         
            (output_feature_tensor_l,
             output_feature_tensor_r) = tf.split(value=output_feature_tensor,
                                                 num_or_size_splits=2,
                                                 axis=0)
            in_label_tensor = tf.placeholder(tf.float16,
                                             [None, len(relevant_variables)],
                                             name="in_label_tensor")
            (in_label_tensor_l,
             in_label_tensor_r) = tf.split(value=in_label_tensor,
                                           num_or_size_splits=2,
                                           axis=0)
            label_weight_tensor = tf.placeholder(tf.float32,
                                             [len(relevant_variables)],
                                             name="label_weight_tensor")
            (loss,
             bool_reduce) = hard_contrastive_loss(
                        feature_tensor_l    = output_feature_tensor_l,
                        feature_tensor_r    = output_feature_tensor_r,
                        input_labels_l      = in_label_tensor_l, 
                        input_labels_r      = in_label_tensor_r,                            
                        label_weight_tensor = label_weight_tensor,
                        margin              = 0.5,
                        similarity_thresh   = similarity_thresh)
            
        elif loss_ind == 'soft_contrastive_incomp_loss':
            # TODO: D4.6: check how often bad combinations are drawn (crossed nan-settings -> Yp=Yn=0)
            (output_feature_tensor_l,
             output_feature_tensor_r) = tf.split(value=output_feature_tensor,
                                                 num_or_size_splits=2,
                                                 axis=0)
            in_label_tensor = tf.placeholder(tf.string,
                                             [None, len(relevant_variables)],
                                             name="in_name_tensor")
            label_weight_tensor = tf.placeholder(tf.float32,
                                             [len(relevant_variables)],
                                             name="label_weight_tensor")
            (in_label_tensor_l,
             in_label_tensor_r) = tf.split(value=in_label_tensor,
                                           num_or_size_splits=2,
                                           axis=0)
            (loss,
             bool_reduce) = soft_contrastive_incomp_loss(
                        feature_tensor_l    = output_feature_tensor_l,
                        feature_tensor_r    = output_feature_tensor_r,
                        input_names_l       = in_label_tensor_l, 
                        input_names_r       = in_label_tensor_r)
        
        else:
            print('no valid key for loss_ind!')
            sys.exit()
            
        # 4.2 Optimizer
        # =============
        with tf.name_scope("train"):
            if optimizer_ind == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer_ind == 'Adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif optimizer_ind == 'GradientDescent':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            if hub_num_retrain == 0:
                grad_var_list = optimizer.compute_gradients(loss,
                                                       tf.trainable_variables())
                for (grad, var) in grad_var_list:
                    retrain_vars
                    tf.summary.histogram(var.name.replace(':', '_') + '/gradient',
                                        grad)
                    tf.summary.histogram(var.op.name, var)

                train_step = optimizer.apply_gradients(grad_var_list)
                variable_list = tf.trainable_variables()
            else:
                retrain_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           scope='CustomLayers'))
                train_step = optimizer.minimize(loss, var_list = retrain_vars)
                variable_list = retrain_vars
            # retrain_vars.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
            #                                             scope='CustomLayers'))
            # train_step = optimizer.minimize(loss, var_list = retrain_vars)
            # variable_list = retrain_vars
            
        # Count number of parameters
        total_parameters = 0
        for variable in variable_list:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
                total_parameters += variable_parameters
        print('Total Number of parameters:', total_parameters) 
        
        # 5. prepare data collections
        # ===========================
        if not val_percentage == 0:
            val_split = int(100/val_percentage)
            assert val_split > 1, "Validation Percentage is too large!"
            print("Validation Percentage: %2.2f%%" % (100/val_split))
            sd_data_dict = subdivide_data_dict(data_dict,
                                               val_split,
                                               loss_ind)
            
            # Choose validation set randomly
            indexlist = np.arange(val_split)
            random.shuffle(indexlist)
            data_dict_val   = sd_data_dict[indexlist[0]]
            train_ = indexlist[1:]
            data_dict_train = {}
            for train_ind in train_:
                for label in sd_data_dict[train_ind].keys():
                    data_dict_train[label] = sd_data_dict[train_ind][label]
                        
        else:
            data_dict_train = data_dict
            data_dict_val = {}
            
        # 6. run training
        # ================================================
        label_tensor = in_label_tensor
        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            # Merge all the summaries and write them out to the logpath
            merged       = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(logpath + 'train', sess.graph)
            val_writer   = tf.summary.FileWriter(logpath + 'val', sess.graph)
            num_train_img = len(list(data_dict_train.keys()))
            num_val_img   = len(list(data_dict_val.keys()))
            train_batch_size = np.minimum(train_batch_size, num_train_img)
            print('number of images in training set:\n', num_train_img)
            print('number of images in validation set:\n', num_val_img)
            
            train_saver = tf.train.Saver()
            training_epoch    = 0
            used_images_train = []
            best_val_loss = None
            for train_iter in range(how_many_training_steps):  
                
                # 6.1 create train batch 
                # ====================== 
                if loss_ind in ['soft_contrastive', 'contrastive_thres','soft_contrastive_incomp_loss']:
                    # To get training samples with multiple labels.
                    # As the batch is split into left and right -> 2*batch_size
                    # Does not consider frac_pos_samples for 'contrastive_thres'
                    (used_images_train,
                     batch_in_img_train,
                     batch_in_label_train,
                     batch_in_names_train) = create_batch(sess,
                                       train_batch_size*2,
                                       data_dict_train,
                                       used_images_train,
                                       jpeg_data_tensor,
                                       in_img_tensor,
                                       label2class_dict,
                                       'soft_triplet')
                else:
                    (used_images_train,
                     batch_in_img_train,
                     batch_in_label_train,
                     batch_in_names_train) = create_batch(sess,
                                       train_batch_size,
                                       data_dict_train,
                                       used_images_train,
                                       jpeg_data_tensor,
                                       in_img_tensor,
                                       label2class_dict,
                                       loss_ind) 
                    
                """EXPERIMENTAL"""
                if not len(np.asarray(batch_in_label_train).shape) == 2:
                    batch_in_label_train = np.expand_dims(batch_in_label_train, -1)
                    batch_in_names_train = np.expand_dims(batch_in_names_train, -1)
                """\EXPERIMENTAL"""

                # Online Data Augmentation
                vardata = [sess.run(augmented_image_tensor, feed_dict={in_img_tensor: imdata}) for imdata in batch_in_img_train]
                batch_in_img_train = vardata


                # 6.2 train architecture + document train steps
                # =============================================
                if len(label_weights) == 0:
                    label_weights = np.ones(len(relevant_variables))
                
                if loss_ind in ['soft_triplet', 'triplet_thresh',
                                  'soft_contrastive', 'contrastive_thres']:
                    feed_dict = {in_batch_tensor:     batch_in_img_train,
                                 label_tensor:        batch_in_label_train,
                                 label_weight_tensor: label_weights,
                                 bool_reduce:         True}
                    
                elif loss_ind in ['soft_contrastive_incomp_loss']:
                    feed_dict = {in_batch_tensor:     batch_in_img_train,
                                 label_tensor:        batch_in_names_train,
                                 label_weight_tensor: label_weights,
                                 bool_reduce:         True}
                    
                elif loss_ind in ['soft_triplet_incomp_loss']:
                    feed_dict = {in_batch_tensor:     batch_in_img_train,
                                 label_tensor:        batch_in_names_train,
                                 label_weight_tensor: label_weights,
                                 bool_reduce:         True}
                    
                (cur_loss, _) = sess.run([loss, train_step],
                                         feed_dict=feed_dict)
                
                print('current train loss (it. '+str(train_iter)+'):\n', cur_loss)
                train_loss = [tf.Summary.Value(tag='loss', simple_value=cur_loss)]
                train_writer.add_summary(tf.Summary(value=train_loss), train_iter)
                train_writer.flush()
                
                # check number of completed training epochs
                if loss_ind in ['contrastive','soft_contrastive',
                                'contrastive_thres','soft_contrastive_incomp_loss']:
                    # needs twice as much images per iteration due to pairs of
                    # samples
                    if (num_train_img-len(used_images_train))/2 < train_batch_size:
                        training_epoch    = training_epoch + 1
                        used_images_train = []
                        print('number of completed training epochs:\n', training_epoch)
                else:
                    if num_train_img-len(used_images_train) < train_batch_size:
                        training_epoch    = training_epoch + 1
                        used_images_train = []
                        print('number of completed training epochs:\n', training_epoch)
                   
                    
                # 6.3 Validate current network weights
                # ====================================    
                if num_val_img > 0 and (train_iter % how_often_validation)==0:
                
                    #     in case of triplet losses, the iterative loss evaluation
                    #     is only an approximation. Not all possible triplets in the
                    #     validation set can be considered in this way to estimate the
                    #     particular triplet loss:(
                    #  -> but it's the best possible option with limited GPU resources 
                    if loss_ind in ['soft_contrastive', 'contrastive_thres','soft_contrastive_incomp_loss']:
                        # To get validation samples with multiple labels.
                        # As the batch is split into left and right -> 2*batch_size
                        # Does not consider frac_pos_samples for 'contrastive_thres'
                        (_,
                         all_batch_in_img_val,
                         all_batch_in_label_val,
                         all_batch_in_names_val ) = create_batch(sess,
                                            -1,
                                            data_dict_val,
                                            [],
                                            jpeg_data_tensor,
                                            in_img_tensor,
                                            label2class_dict,
                                            'soft_triplet')
                        # needs an even number of samples per batch
                        # -> adapt train_batch_size so that in iterative validation an
                        #    even number is guaranteed
                        train_batch_size_old = train_batch_size
                        if train_batch_size % 2 != 0:
                            train_batch_size_old = train_batch_size
                            train_batch_size -= 1
                    else:
                        (_,
                         all_batch_in_img_val,
                         all_batch_in_label_val,
                         all_batch_in_names_val ) = create_batch(sess,
                                            -1,
                                            data_dict_val,
                                            [],
                                            jpeg_data_tensor,
                                            in_img_tensor,
                                            label2class_dict,
                                            loss_ind)
                        
                    losses_val = []
                    losses_sum = 0
                    num_positive_triplets = 0
                    total_num_val = num_val_img
                    
                    for temp_val in range(math.ceil(total_num_val/train_batch_size)):
                        if (temp_val+1)*train_batch_size < len(all_batch_in_label_val):
                            batch_in_img_val   = all_batch_in_img_val[temp_val*train_batch_size:(temp_val+1)*train_batch_size]
                            batch_in_label_val = all_batch_in_label_val[temp_val*train_batch_size:(temp_val+1)*train_batch_size]
                            batch_in_names_val = all_batch_in_names_val[temp_val*train_batch_size:(temp_val+1)*train_batch_size]
                        else:
                            batch_in_img_val   = all_batch_in_img_val[temp_val*train_batch_size::]
                            batch_in_label_val = all_batch_in_label_val[temp_val*train_batch_size::]
                            batch_in_names_val = all_batch_in_names_val[temp_val*train_batch_size::] 
                            if loss_ind in ['soft_contrastive', 'contrastive_thres','soft_contrastive_incomp_loss']\
                            and np.shape(batch_in_label_val)[0] % 2 != 0:
                                # needs an even number of samples per batch
                                batch_in_img_val   = batch_in_img_val[0:-1]
                                batch_in_label_val = batch_in_label_val[0:-1]
                                batch_in_names_val = batch_in_names_val[0:-1]
                                
                        if len(batch_in_img_val) == 0 or \
                        (len(batch_in_img_val) < 3 and loss_ind=='soft_triplet'):
                            break
                        
                        """EXPERIMENTAL"""
                        if not len(np.asarray(batch_in_label_val).shape) == 2:
                            batch_in_label_val = np.expand_dims(batch_in_label_val, -1)
                            batch_in_names_val = np.expand_dims(batch_in_names_val, -1)
                        """\EXPERIMENTAL"""
                        
                        # prepare dicts for running the session   
                        if loss_ind in ['soft_triplet', 'triplet_thresh',
                                      'soft_contrastive', 'contrastive_thres']:
                            feed_dict = {in_batch_tensor: batch_in_img_val,
                                         label_tensor:    batch_in_label_val,
                                         label_weight_tensor: label_weights,
                                         bool_reduce:     False}
                    
                        elif loss_ind in ['soft_contrastive_incomp_loss']:
                            feed_dict = {in_batch_tensor:     batch_in_img_val,
                                         label_tensor:        batch_in_names_val,
                                         label_weight_tensor: label_weights,
                                         bool_reduce:         False}
                        
                        elif loss_ind in ['soft_triplet_incomp_loss']:
                            feed_dict = {in_batch_tensor:     batch_in_img_val,
                                         label_tensor:        batch_in_names_val,
                                         label_weight_tensor: label_weights,
                                         bool_reduce:         False}
                            
                                
                        (temp_losses_val) = sess.run([loss],
                                                     feed_dict=feed_dict)
                        
                        # losses_val is not empty anymore
                        if np.shape(losses_val)[0] > 0:
                            # needs more than one entry per temp_losses_val for
                            # concatenating
                            # -> is interesting in the case that a "rest" (< train_batch_size)
                            #    of validation samples was estimated
                            # -> in case of triplet losses, >=3 are needed
                            #    (implies >=3 samples to enable a triplet)
                            if len(np.shape(np.squeeze(temp_losses_val))) > 0\
                            and loss_ind not in ['triplet_loss', 'soft_triplet',
                                                 'triplet_thresh','soft_triplet_incomp_loss']:
                                losses_val = np.concatenate((losses_val,
                                                             np.squeeze(temp_losses_val)), axis=0)
                            else:
                                if loss_ind in ['triplet_loss', 'soft_triplet',
                                                 'triplet_thresh','soft_triplet_incomp_loss']:
                                    # at least 3 entries needed
                                    if np.shape(np.squeeze(temp_losses_val))[0] >= 3:
                                        # output is [bathc_size, batch_size, bacth_size]
                                        # -> needs special handling
                                        if loss_ind in ['triplet_loss', 'triplet_thresh']\
                                        or loss_ind in ['soft_triplet','soft_triplet_incomp_loss']:
                                            losses_sum            += np.sum(temp_losses_val)
                                            num_positive_triplets += np.sum((np.asarray(temp_losses_val) > 1e-16).astype(float))
                                        # output is [batch_size]
                                        else:
                                            losses_val = np.concatenate((losses_val,
                                                                 np.squeeze(temp_losses_val)), axis=0)
                                    # less than 3 samples do not deliver a reasonable loss!
                                    else:
                                        continue
                                # no triplet loss and only one loss value was estimated
                                # -> needs another form of concatenating
                                else:
                                    losses_val = np.concatenate((losses_val,
                                                                 temp_losses_val[0][0]), axis=0)
                        # losses_val is empty by now
                        # ignore the case that less than 3 samples occur in this case
                        # means that the train_batch_size is assumed to be at least 3
                        else:
                            losses_val = np.squeeze(temp_losses_val)  
                    
                    if loss_ind in ['soft_contrastive', 'contrastive_thres']:
                        if train_batch_size_old % 2 != 0:
                            train_batch_size = train_batch_size_old
                    
                    if loss_ind in ['triplet_loss', 'triplet_thresh']\
                    or loss_ind in ['soft_triplet','soft_triplet_incomp_loss']:
                        # build the mean only amon valid triplets            
                        cur_loss_val = losses_sum/(num_positive_triplets + 1e-16)
        #                valid_triplets        = (losses_val > 1e-16).astype(float)
        #                num_positive_triplets = np.sum(valid_triplets)
        #                cur_loss_val = np.sum(losses_val)/(num_positive_triplets + 1e-16)
                    else:
                        cur_loss_val = np.mean(losses_val)
                    
                    print('current val loss (it. '+str(train_iter)+'):\n', cur_loss_val)
                    val_loss = [tf.Summary.Value(tag='loss', simple_value=cur_loss_val)]
                    val_writer.add_summary(tf.Summary(value=val_loss), train_iter)
                    val_writer.flush()
                    
                    # Early Stopping
                    if best_val_loss is None or cur_loss_val < best_val_loss:
                        print("New best model found!")
                        train_saver.save(sess, logpath+r"/"+CHECKPOINT_NAME)
                        best_val_loss = cur_loss_val


if __name__ == "__main__":
    evaluate_model(r"../", r"../")