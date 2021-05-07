# import silk_retrieval_class as src
import sys

# sys.path.insert(0, r"./src/")
# import DatasetCreation
from . import DatasetCreation
from . import silk_retrieval_class as src


def create_dataset_parameter(csvfile,
                             imgsavepath,
                             master_file_dir,
                             masterfileRules,
                             minnumsamples=150,
                             retaincollections=['cer', 'garin', 'imatex', 'joconde', 'mad', 'met',
                                                'mfa', 'mobilier', 'mtmad', 'paris-musees', 'risd',
                                                'smithsonian', 'unipa', 'vam', 'venezia', 'versailles'],
                             num_labeled=1,
                             flagRuleDataset=True,
                             multi_label_variables=["material"]):
    r"""Sets all dataset utilities up.

    :Arguments\::
        :csvfile (*string*)\::
            The name (including the path) of the CSV file containing the data exported from the
            SILKNOW knowledge graph.
        :imgsavepath (*string*)\::
            The path to the directory that will contain the downloaded images. The original images
            will be downloaded to the folder img_unscaled in that directory and the rescaled images
            (the smaller side will be 448 pixels) will be saved to the folder img. It has to be relative
            to the current working directory.
        :minnumsaples (*int*)\::
            The minimum number of samples that has to be available for a single class or, in case the parameter
            multi_label_variables is not None, for every class combination for the variables contained in that list.
            The dataset is restricted to class combinations that occur at least minnumsamples times in the dataset
            exported from the knowledge graph. Classes or class combinations with fewer samples will not be
            considered in the generated dataset.
        :retaincollections (*list*)\::
            A list containing the museums/collections in the knowledge graph that
            shall be considered for the data set creation. Data from museums/collections
            not stated in this list will be omitted. Possible values in the list according
            to EURECOM’s export from the SILKNOW knowledge graph are: 'cer', 'garin',
            'imatex', 'joconde', 'mad', 'met', 'mfa', 'mobilier', 'mtmad', 'paris-musees',
            'risd', 'smithsonian', 'unipa', 'vam', 'venezia', 'versailles'.
        :num_labeled (*int*)\::
            A variable that indicates how many labels per sample should be available so that a sample is a valid
            sample and thus, part of the created dataset. The maximum value is 5, as five semantic variables
            are considered in the current implementation of this function. Choosing this maximum number means
            that only complete samples will form the dataset, while choosing a value of 0 means that records
            without annotations will also be considered. The value of num_labeled must not be smaller than 0.
        :master_file_dir (*string*)\::
            Directory where the collection files and masterfile will be created. The storage location can now
            be chosen by the user.
        :flagRuleDataset (*bool*)\::
            Boolean variable indicating whether the rule subset shall be generated (True) or not (False).
        :masterfileRules (*string*)\::
            Name of the rule master file that is assumed to be available in master_file_dir. The rule master
            file lists all rule files, each of which contains a list of the URIs of all objects which are
            considered to be similar or dissimilar according to one of the domain experts’ rules.
        :multi_label_variables (*list of strings*)\::


    :Returns\::
        No returns. This function produces all data files needed for running the other functions
        in this python package.
    """

    DatasetCreation.createDatasetForCombinedSimilarityLoss(rawCSVFile=csvfile,
                                                           imageSaveDirectory=imgsavepath,
                                                           masterfileDirectory=master_file_dir,
                                                           minNumSamplesPerClass=minnumsamples,
                                                           retainCollections=retaincollections,
                                                           minNumLabelsPerSample=num_labeled,
                                                           flagDownloadImages=True,
                                                           flagRescaleImages=True,
                                                           flagRuleDataset=flagRuleDataset,
                                                           masterfileRules=masterfileRules,
                                                           flagColourAugmentDataset=False,
                                                           multiLabelsListOfVariables=multi_label_variables)


def train_model_parameter(master_file_name,
                          master_file_dir,
                          master_file_rules_name,
                          master_file_similar,
                          master_file_dissimilar,
                          log_dir,
                          model_dir,
                          add_fc=[1024, 128],
                          num_fine_tune_layers=0,
                          batch_size=150,
                          rules_batch_fraction=0.5,
                          how_many_training_steps=300,
                          learning_rate=1e-4,
                          validation_percentage=25,
                          how_often_validation=10,
                          loss_ind="combined_similarity_loss",
                          loss_weight_semantic= 1 / 4,
                          loss_weight_rules= 1 / 4,
                          loss_weight_colour= 1 / 4,
                          loss_weight_augment= 1 / 4,
                          variable_weights=[0.3, 0.25, 0.2, 0.15, 0.1],
                          relevant_variables=["depiction", "material", "place", "technique", "timespan"],
                          random_crop=[0.7, 1.0],
                          random_rotation90=True,
                          gaussian_noise=0.1,
                          flip_left_right=True,
                          flip_up_down=True,
                          multi_label_variables=["material"]):
    r""" Trains a new model for calculating feature vectors.

    :Arguments\::
        :master_file_name (*string*)\::
            Name of the master file that lists the collection files with the available samples of the
            labelled subset. This file has to exist in directory master_file_dir.
        :master_file_dir (*string*)\::
            Path to the directory containing the master file.
        :master_file_rules_name (*string*)\::
            The name of the master file listing the collection files "collection_rules*.txt" containing the
            samples of the rule subset that are not contained in the labelled subset. These files have to exist
            in directory master_file_name. The image paths listed in the collection files have to be given relative
            to master_file_name. This parameter is only required if rules_batch_fraction > 0.
        :master_file_similar (*string*)\::
            The name of the master file listing the rule files, each of which contains a list of URIs of all objects
            considered to be similar according to one of the rules defined inTable 8. See also the description of
            parameter masterfileRules in Table 9. This parameter is only required if rules_batch_fraction > 0.
        :master_file_dissimilar (*string*)\::
            The name of the master file listing the rule files, each of which contains a list of URIs of all objects
            considered to be dissimilar. The rules defined for master_file_similar also apply.
        :log_dir (*string*):
            Path to the directory to which the log files will be saved.
        :model_dir (*string*):
            Path to the directory to which the trained model will be saved.
        :batch_size (*int*)\::
            Number of samples that are used during each training iteration.
        :how_many_training_steps (*int*)\::
            Number of training iterations.
        :learning_rate (*float*)\::
            Learning rate for the training procedure.
        :add_fc (*array of int*)\::
            The number of fully connected (fc) layers to be trained on top of the ResNet is equal to the number of
            entries in the array. Each entry is an int specifying the number of nodes in the individual fc layers.
            If [1000, 100] is given, two fc layers will be added, the first having 1000 and the second having 100
            nodes. If no layers should be added, an empty array '[]' has to be given.
        :num_fine_tune_layers (*int*)\::
            Number of residual blocks (each containing three convolutional layers) of ResNet 152 that shall be
            retrained.
        :loss_ind (*string*)\::
            The loss function that shall be minimized to train the network. Possible values: 'triplet',
            'colour', 'rule', 'combined_similarity_loss'.
        :loss_weight_semantic (*float*)\::
            Weight (float in [0, 1]) for the semantic similarity term in the combined loss.
        :loss_weight_rules (*float*)\::
            Weight (float in [0, 1]) for the rules similarity term in the combined loss.
        :loss_weight_colour (*float*)\::
            Weight (float in [0, 1]) for the colour similarity term in the combined loss.
        :loss_weight_augment (*float*)\::
            Weight (float in [0, 1]) for the self-similarity term in the combined loss.
        :relevant_variables (*list*)\::
            A list containing the names of the variables to be considered in the definition of semantic similarity.
            These names have to be those (or a subset of those) listed in the header sections of the collection
            files collection_n.txt.
        :variable_weights (*list of floats*)\::
            List of weights (floats in [0, 1]) describing the importance of the semantic variables in the
            definition of semantic similarity. The order has to be the same as in relevant_variables and the
            sum has to be 1.
        :how_often_validation (*int*)\::
            Number of training iterations between two computations of the validation loss.
        :validation_percentage (*int*)\::
            Percentage of training samples that are used for validation. The value has to be in the range [0, 100).
        :random_crop (*list*)\::
            Range of float fractions for centrally cropping the image. The crop fraction
            is drawn out of the provided range [lower bound, upper bound],
            i.e. the first and second values of random_crop. If [0.8, 0.9] is given,
            a crop fraction of e.g. 0.85 is drawn meaning that the crop for an image with
            the dimensions 200 x 400 pixels consists of the 170 x 340 central pixels.
        :random_rotation90 (*bool*)\::
            Data augmentation: should rotations by 90° be used (True) or not (False)?
        :gaussian_noise (*float*)\::
            Data augmentation: Standard deviation of the Gaussian noise
        :flip_left_right (*bool*)\::
            Data augmentation: should horizontal flips be used (True) or not (False)?
        :flip_up_down (*bool*)\::
            Data augmentation: should vertical flips be used (True) or not (False)?.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the five semantic variables (relevant_variables) that have multiple class
            labels per variable to be used. A complete list would be ["material", "place", "timespan",
            "technique", "depiction"].
        :rules_batch_fraction (*float*)\::
            Fraction of images in a minibatch (float in [0, 1] to be drawn from the images without annotations
            in the rules subset.

    :Returns\::
        No returns. The trained graph (containing the pre-trained ResNet 152 and the
        trained new layers) is stored automatically in the directory given in
        the variable model_dir and the according training summary in log_dir.

    """

    # create new retrieval object
    sr = src.SilkRetriever()

    # set parameters
    sr.master_file_name = master_file_name
    sr.master_file_dir = master_file_dir
    sr.log_dir = log_dir
    sr.model_dir = model_dir

    sr.master_file_rules_name = master_file_rules_name
    sr.master_file_similar_obj = master_file_similar
    sr.master_file_dissimilar_obj = master_file_dissimilar

    sr.add_fc = add_fc
    sr.num_fine_tune_layers = num_fine_tune_layers

    sr.batch_size = batch_size
    sr.how_many_training_steps = how_many_training_steps
    sr.learning_rate = learning_rate
    sr.validation_percentage = validation_percentage
    sr.how_often_validation = how_often_validation

    sr.relevant_variables = relevant_variables
    sr.variable_weights = variable_weights

    if loss_ind == "combined_similarity_loss":
        sr.loss_ind = loss_ind
        sr.loss_weight_semantic = loss_weight_semantic
        sr.loss_weight_rules = loss_weight_rules
        sr.loss_weight_colour = loss_weight_colour
        sr.loss_weight_augment = loss_weight_augment
    else:
        sr.loss_ind = "combined_similarity_loss"
        sr.loss_weight_semantic = 0.
        sr.loss_weight_rules = 0.
        sr.loss_weight_colour = 0.
        sr.loss_weight_augment = 0.
        if loss_ind == "triplet":
            sr.loss_weight_semantic = 1.
        elif loss_ind == "colour":
            sr.loss_weight_colour = 1.
        elif loss_ind == "rule":
            sr.loss_weight_rules = 1.
        elif loss_ind == "self_augment":
            sr.loss_weight_augment = 1.

    sr.semantic_batch_size_fraction = 1-rules_batch_fraction
    sr.rules_batch_size_fraction = rules_batch_fraction
    sr.colour_augment_batch_size_fraction = 0

    sr.aug_set_dict['random_crop'] = random_crop
    sr.aug_set_dict['random_rotation90'] = random_rotation90
    sr.aug_set_dict['gaussian_noise'] = gaussian_noise
    sr.aug_set_dict['flip_left_right'] = flip_left_right
    sr.aug_set_dict['flip_up_down'] = flip_up_down

    sr.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sr.train_model()


def build_kDTree_parameter(model_dir,
                           master_file_tree,
                           master_dir_tree,
                           tree_dir,
                           relevant_variables=["depiction", "material", "place", "technique", "timespan"],
                           multi_label_variables=["material"]):
    r"""Builds a kD-Tree using a pre-trained network.

    :Arguments\::
        :model_dir (*string*):
            Path (without filename) to a folder containing a pre-trained network. Only one pre-trained model
            should be stored in that model folder.
        :master_file_tree (*string*)\::
            Name of the master file listing the collection files with the available samples. It has to exist in
            the directory master_dir_tree.
        :master_dir_tree (*string*)\::
            Path to the directory containing the master file master_file_tree.
        :relevant_variables (*list*)\::
            A list containing the names of the variables to be considered in the definition of semantic similarity.
            These names have to be those used at training time of the CNN model passed to model_dir.
        :tree_dir (*string*):
            Path to where the kD-tree "kdtree.npz" will be saved. It's a dictionary containing
            the following key value pairs:
                - Tree: The kD-tree accroding to the python toolbox sklearn.neighbors.
                - Labels: The class label indices of the data whose feature vectors are stored in the tree nodes.
                - DictTrain: The image data including their labels that was used to build the tree.
                - relevant_variables: A list of the image data's labels that shall be considered.
                - label2class_list: A list assigning the label indices to label names.
        :multi_label_variables (*list of strings*)\::
            A list of names of those variables (relevant_variables) that have multiple class labels per sample.
            A complete list would be ["material", "place", "timespan", "technique", "depiction"].

    :Returns\::
        No returns. The kD-tree is stored automatically in the directory given in
        the variable tree_dir.
    """

    # create new retrieval object
    sr = src.SilkRetriever()

    # set parameters
    sr.model_dir = model_dir
    sr.master_file_tree = master_file_tree
    sr.master_dir_tree = master_dir_tree
    sr.tree_dir = tree_dir
    sr.relevant_variables = relevant_variables
    sr.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sr.build_kd_tree()


def get_kNN_parameter(tree_dir,
                      master_file_retrieval,
                      master_dir_retrieval,
                      model_dir,
                      pred_gt_dir,
                      num_neighbors=10,
                      bool_labeled_input=False,
                      multi_label_variables=["material"]):
    r"""Retrieves the k nearest neighbours from a given kdTree.

    :Arguments\::
        :tree_dir (*string*)\::
            Path to where the kD-tree "kdtree.npz" was saved. It's a dictionary containing
            the following key value pairs:
                - Tree: The kD-tree accroding to the python toolbox sklearn.neighbors.
                - Labels: The class label indices of the data whose feature vectors are stored in the tree nodes.
                - DictTrain: The image data including their labels that was used to build the tree.
                - relevant_variables: A list of the image data's labels that shall be considered.
                - label2class_list: A list assigning the label indices to label names.
        :master_file_retrieval (*string*)\::
            Name of the master file that lists the collection files with the available samples. This file has to
            exist in directory master_dir_retrieval.
        :master_dir_retrieval (*string*)\::
            Path to the directory containing the master file master_file_retrieval.
        :bool_labeled_input (*bool*)\::
            An indicator whether annotations are provided in the collection files that are listed in the master file
            for the input samples. Thus, this parameter defines the use case for the image retrieval.
            If bool_labeled_input = False, the query images are assumed to be unlabeled, as it is the standard
            situation in the SILKNOW use case. Otherwise, the search images are assumed to have class labels, as
            required for images to be used for evaluation.
        :model_dir (*string*):
            Path (without filename) to a folder containing a pre-trained network. Only one pre-trained model should
            be stored in that model folder.
        :num_neighbors (*int*)\::
            Number of nearest neighbours that are retrieved from a kD-Tree.
        :pred_gt_dir (*string*):
            Path to where the results will be saved.
            In case of provided labeled data it's a dictionary "pred_gt.npz" containing the following key value pairs:
                - Groundtruth: The label indicies of the provided data.
                - Predictions: The label indices of the found k nearest neighbours.
                - label2class_list: A list assigning the label indices to label names.
            as well as text file "knn_list.txt" containing the predicted and ground truth labels.
            In case of provided unlabeled data it's a text file "knn_list.txt" containing only the predictions.
            Furthermore, a CSV-file "kNN_LUT.csv" is created in any case containing the values:
                - input_image_name: The filename of the input image (including the path).
                - kNN_image_names: The full image names of the kNN.
                - kNN_kg_object_uri: The knowledge graph object URIs of the kNNs.
                - kNN_kD_index: The indices of the kNN of the input image in the kD-tree.
                - kNN_descriptor_dist: The distances of the descriptors of the kNN to the descriptor of the input image.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the five semantic variables (relevant_variables) that have multiple class
            labels per variable. A complete list would be ["material", "place", "timespan", "technique", "depiction"].
            A multi-label kNN-classification is realised for the list of variables indicated by this parameter instead
            of the default single-label kNN-classification.

    :Returns\::
        No returns. The result is stored automatically in the directory given in
        the variable pred_gt_dir.
    """

    sr = src.SilkRetriever()

    # set parameters
    sr.tree_dir = tree_dir
    sr.master_file_retrieval = master_file_retrieval
    sr.master_dir_retrieval = master_dir_retrieval
    sr.model_dir = model_dir
    sr.pred_gt_dir = pred_gt_dir
    sr.num_neighbors = num_neighbors
    sr.bool_labeled_input = bool_labeled_input
    sr.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sr.get_knn()


def evaluate_model_parameter(pred_gt_dir,
                             eval_result_dir,
                             multi_label_variables=["material"]):
    r""" Evaluates a pre-trained model.

    :Arguments\::
        :pred_gt_path (*string*)\::
            Path to a directory containing the file pred_gt.npz containing the output of the function
            get_kNN_parameter. This file contains the input of the evaluation procedure.
        :eval_result_dir (*string*)\::
            Path to where the evaluation results will be saved. The evaluation results are stored in multiple
            files per variable.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the five semantic variables (relevant_variables) that have multiple class
            labels per variable. A complete list would be ["material", "place", "timespan", "technique", "depiction"].

    :Returns\::
        No returns. The results are stored automatically in the directory given in
        the variable eval_result_dir.
    """

    # create new retrieval object
    sr = src.SilkRetriever()

    # set parameters
    sr.pred_gt_dir = pred_gt_dir
    sr.eval_result_dir = eval_result_dir
    sr.multiLabelsListOfVariables = multi_label_variables

    # call main function
    sr.evaluate_model()


def cross_validation_parameter(master_file_name,
                               master_file_dir,
                               master_file_rules_name,
                               master_file_similar,
                               master_file_dissimilar,
                               log_dir,
                               model_dir,
                               tree_dir,
                               pred_gt_dir,
                               eval_result_dir,
                               add_fc=[1024, 128],
                               num_fine_tune_layers=0,
                               batch_size=150,
                               rules_batch_fraction=0.5,
                               how_many_training_steps=300,
                               learning_rate=1e-4,
                               validation_percentage=25,
                               how_often_validation=10,
                               loss_ind="combined_similarity_loss",
                               num_neighbors=10,
                               variable_weights=[0.3, 0.25, 0.2, 0.15, 0.1],
                               relevant_variables=["depiction", "material", "place", "technique", "timespan"],
                               loss_weight_semantic=1 / 4,
                               loss_weight_rules=1 / 4,
                               loss_weight_colour=1 / 4,
                               loss_weight_augment=1 / 4,
                               multi_label_variables=["material"],
                               random_crop=[0.7, 1.0],
                               random_rotation90=True,
                               gaussian_noise=0.1,
                               flip_left_right=True,
                               flip_up_down=True,):
    r""" Performs 5-fold cross validation.

    :Arguments\::
        :master_file_name (*string*)\::
            Name of the master file that lists the collection files with the available samples of the
            labelled subset. This file has to exist in directory master_file_dir.
        :master_file_dir (*string*)\::
            Path to the directory containing the master file.
        :master_file_rules_name (*string*)\::
            The name of the master file listing the collection files "collection_rules*.txt" containing the
            samples of the rule subset that are not contained in the labelled subset. These files have to exist
            in directory master_file_name. The image paths listed in the collection files have to be given relative
            to master_file_name. This parameter is only required if rules_batch_fraction > 0.
        :master_file_similar (*string*)\::
            The name of the master file listing one file per rule that describe similar objects only.
            Each listed file contains one object URI per line of an object in the SILKNOW knowledge graph
            that contributes to the respective "similar rule".
        :master_file_dissimilar (*string*)\::
            The name of the master file listing one file per rule that describe dissimilar objects only.
            Each listed file contains one object URI per line of an object in the SILKNOW knowledge graph
            that contributes to the respective "dissimilar rule".
        :log_dir (*string*):
            Path to the directory to which the log files will be saved.
        :model_dir (*string*):
            Path to the directory to which the trained model will be saved.
        :tree_dir (*string*):
            Path to where the kD-tree "kdtree.npz" will be saved. It's a dictionary containing
            the following key value pairs:
                - Tree: The kD-tree accroding to the python toolbox sklearn.neighbors.
                - Labels: The class label indices of the data whose feature vectors are stored in the tree nodes.
                - DictTrain: The image data including their labels that was used to build the tree.
                - relevant_variables: A list of the image data's labels that shall be considered.
                - label2class_list: A list assigning the label indices to label names.
        :num_neighbors (*int*)\::
            Number of nearest neighbours that are retrieved from a kD-Tree.
        :pred_gt_dir (*string*):
            Path to where the results will be saved.
            In case of provided labeled data it's a dictionary "pred_gt.npz" containing the following key value pairs:
                - Groundtruth: The label indicies of the provided data.
                - Predictions: The label indices of the found k nearest neighbours.
                - label2class_list: A list assigning the label indices to label names.
            as well as text file "knn_list.txt" containing the predicted and ground truth labels.
            In case of provided unlabeled data it's a text file "knn_list.txt" containing only the predictions.
            Furthermore, a CSV-file "kNN_LUT.csv" is created in any case containing the values:
                - input_image_name: The filename of the input image (including the path).
                - kNN_image_names: The full image names of the kNN.
                - kNN_kg_object_uri: The knowledge graph object URIs of the kNNs.
                - kNN_kD_index: The indices of the kNN of the input image in the kD-tree.
                - kNN_descriptor_dist: The distances of the descriptors of the kNN to the descriptor of the input image.
        :eval_result_dir (*string*)\::
            Path to where the evaluation results will be saved. The evaluation results are stored in multiple
            files per variable.
        :batch_size (*int*)\::
            Number of samples that are used during each training iteration.
        :how_many_training_steps (*int*)\::
            Number of training iterations.
        :learning_rate (*float*)\::
            Learning rate for the training procedure.
        :add_fc (*array of int*)\::
            The number of fully connected (fc) layers to be trained on top of the ResNet is equal to the number of
            entries in the array. Each entry is an int specifying the number of nodes in the individual fc layers.
            If [1000, 100] is given, two fc layers will be added, the first having 1000 and the second having 100
            nodes. If no layers should be added, an empty array '[]' has to be given.
        :num_fine_tune_layers (*int*)\::
            Number of residual blocks (each containing three convolutional layers) of ResNet 152 that shall be
            retrained.
        :loss_ind (*string*)\::
            The loss function that shall be minimized to train the network. Possible values: 'triplet',
            'colour', 'rule', 'combined_similarity_loss'.
        :loss_weight_semantic (*float*)\::
            Weight (float in [0, 1]) for the semantic similarity term in the combined loss.
        :loss_weight_rules (*float*)\::
            Weight (float in [0, 1]) for the rules similarity term in the combined loss.
        :loss_weight_colour (*float*)\::
            Weight (float in [0, 1]) for the colour similarity term in the combined loss.
        :loss_weight_augment (*float*)\::
            Weight (float in [0, 1]) for the self-similarity term in the combined loss.
        :relevant_variables (*list*)\::
            A list containing the names of the variables to be considered in the definition of semantic similarity.
            These names have to be those (or a subset of those) listed in the header sections of the collection
            files collection_n.txt.
        :variable_weights (*list of floats*)\::
            List of weights (floats in [0, 1]) describing the importance of the semantic variables in the
            definition of semantic similarity. The order has to be the same as in relevant_variables and the
            sum has to be 1.
        :how_often_validation (*int*)\::
            Number of training iterations between two computations of the validation loss.
        :validation_percentage (*int*)\::
            Percentage of training samples that are used for validation. The value has to be in the range [0, 100).
        :random_crop (*list*)\::
            Range of float fractions for centrally cropping the image. The crop fraction
            is drawn out of the provided range [lower bound, upper bound],
            i.e. the first and second values of random_crop. If [0.8, 0.9] is given,
            a crop fraction of e.g. 0.85 is drawn meaning that the crop for an image with
            the dimensions 200 x 400 pixels consists of the 170 x 340 central pixels.
        :random_rotation90 (*bool*)\::
            Data augmentation: should rotations by 90° be used (True) or not (False)?
        :gaussian_noise (*float*)\::
            Data augmentation: Standard deviation of the Gaussian noise
        :flip_left_right (*bool*)\::
            Data augmentation: should horizontal flips be used (True) or not (False)?
        :flip_up_down (*bool*)\::
            Data augmentation: should vertical flips be used (True) or not (False)?.
        :multi_label_variables (*list of strings*)\::
            A list of variable names of the five semantic variables (relevant_variables) that have multiple class
            labels per variable to be used. A complete list would be ["material", "place", "timespan",
            "technique", "depiction"].
        :rules_batch_fraction (*float*)\::
            Fraction of images in a minibatch (float in [0, 1] to be drawn from the images without annotations
            in the rules subset.

    :Returns\::
        No returns; all results will be stored automatically. The training summary in log_dir, the trained models of
        cross validation iterations in model_dir, the according kD-trees is tree_dir, the results of the image
        retrieval in pred_gt_dir and the results of the evalutaion in eval_result_dir.

    """

    # create new retrieval object
    sr = src.SilkRetriever()

    # set parameters
    sr.master_file_name_cv = master_file_name
    sr.master_file_rules_name = master_file_rules_name
    sr.master_file_rules_name_cv = master_file_rules_name
    sr.master_file_similar_obj = master_file_similar
    sr.master_file_dissimilar_obj = master_file_dissimilar

    sr.master_file_dir_cv = master_file_dir
    sr.log_dir_cv = log_dir
    sr.model_dir_cv = model_dir
    sr.tree_dir_cv = tree_dir
    sr.pred_gt_dir_cv = pred_gt_dir
    sr.eval_result_dir_cv = eval_result_dir

    sr.master_file_dir = master_file_dir
    sr.log_dir = log_dir
    sr.model_dir = model_dir
    sr.tree_dir = tree_dir
    sr.pred_gt_dir = pred_gt_dir
    sr.eval_result_dir = eval_result_dir

    sr.add_fc = add_fc
    sr.num_fine_tune_layers = num_fine_tune_layers

    sr.batch_size = batch_size
    sr.how_many_training_steps = how_many_training_steps
    sr.learning_rate = learning_rate
    sr.validation_percentage = validation_percentage
    sr.how_often_validation = how_often_validation

    sr.relevant_variables = relevant_variables
    sr.variable_weights = variable_weights
    sr.multiLabelsListOfVariables = multi_label_variables

    sr.num_neighbors = num_neighbors

    if loss_ind == "combined_similarity_loss":
        sr.loss_ind = loss_ind
        sr.loss_weight_semantic = loss_weight_semantic
        sr.loss_weight_rules = loss_weight_rules
        sr.loss_weight_colour = loss_weight_colour
        sr.loss_weight_augment = loss_weight_augment
    else:
        sr.loss_ind = "combined_similarity_loss"
        sr.loss_weight_semantic = 0.
        sr.loss_weight_rules = 0.
        sr.loss_weight_colour = 0.
        sr.loss_weight_augment = 0.
        if loss_ind == "triplet":
            sr.loss_weight_semantic = 1.
        elif loss_ind == "colour":
            sr.loss_weight_colour = 1.
        elif loss_ind == "rule":
            sr.loss_weight_rules = 1.
        elif loss_ind == "self_augment":
            sr.loss_weight_augment = 1.

    sr.semantic_batch_size_fraction = 1 - rules_batch_fraction
    sr.rules_batch_size_fraction = rules_batch_fraction
    sr.colour_augment_batch_size_fraction = 0

    sr.aug_set_dict['random_crop'] = random_crop
    sr.aug_set_dict['random_rotation90'] = random_rotation90
    sr.aug_set_dict['gaussian_noise'] = gaussian_noise
    sr.aug_set_dict['flip_left_right'] = flip_left_right
    sr.aug_set_dict['flip_up_down'] = flip_up_down

    # call main function
    sr.cross_validation()
