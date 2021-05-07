# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
import sys
# import random
import tensorflow_hub as hub
# import tensorflow_probability as tfp
import urllib
import pandas as pd
import urllib.request
import math
import matplotlib.pyplot as plt
import cv2
import collections
# import timeit

from tqdm import tqdm
from sklearn.neighbors import KDTree
from operator import itemgetter
from scipy import stats
from shutil import copy

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# sys.path.insert(0, '../')
# import SILKNOW_WP4_library as wp4lib
# import hue_saturation_analysis as hue_sat
# from SampleHandler import SampleHandler
# from SimilarityLosses import SimilarityLosses

from . import SILKNOW_WP4_library as wp4lib
from . import SampleHandler
from . import SimilarityLosses
from . import hue_saturation_analysis as hue_sat
import time


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
        self.sess = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + 'model.ckpt' + '.meta',
                                               clear_devices=True)
            """EXPERIMENTELL"""
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)
            #            self.sess.run(tf.local_variables_initializer())
            """EXPERIMENTELL"""
            saver.restore(self.sess, loc + 'model.ckpt')
            # self.output_features = self.graph.get_operation_by_name('CustomLayers/output_features').outputs[0]
            self.output_features = self.graph.get_operation_by_name('normalized_descriptor').outputs[0]

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
        feed_dict_raw = {"DecodeJPGInput:0": data}
        decoded_img_op = self.graph.get_operation_by_name('Squeeze').outputs[0]
        decoded_img = self.sess.run(decoded_img_op, feed_dict=feed_dict_raw)
        decoded_img = np.expand_dims(np.asarray(decoded_img), 0)

        feed_dict_decoded = {"ModuleLayers/input_img:0": decoded_img}
        output = self.sess.run(self.output_features, feed_dict=feed_dict_decoded)
        output = np.squeeze(output)
        return output


class SilkRetriever:
    """Class for handling all functions of the classifier."""

    def __init__(self):
        """Creates an empty object of class silk_classifier."""
        """------------------ Create data set -----------------------------------------------------------------------"""
        self.csv_file = ""
        self.img_save_dir = ""
        self.min_samples_class = ""
        self.retain_collections = ""
        self.num_labeled = 1
        self.download_images = None
        self.rescale_images = None

        self.multiLabelsListOfVariables = None

        """------------------ Directories ---------------------------------------------------------------------------"""
        # data for training
        self.master_file_name = ""
        self.master_file_name_cv = ""
        self.master_file_dir = ""

        # additional training data for rules (only for combined loss)
        self.master_file_rules_name = ""
        self.master_file_rules_name_cv = ""
        self.master_file_similar_obj = ""
        self.master_file_dissimilar_obj = ""

        # additional training data for colour and augment only (only for combined loss)
        self.master_file_colour_augment_name = ""
        self.master_file_colour_augment_name_cv = ""

        # data for tree
        self.master_file_tree = ""
        self.master_file_tree_cv = ""
        self.master_dir_tree = ""

        # data for retrieval
        self.master_file_retrieval = ""
        self.master_file_retrieval_cv = ""
        self.master_dir_retrieval = ""
        self.master_file_retrieval_rules = ""
        self.bool_in_cv = False

        # save training
        self.log_dir = ""
        self.log_dir_cv = ""
        self.model_dir = ""
        self.model_dir_cv = ""

        # save tree
        self.tree_dir = ""
        self.tree_dir_cv = ""

        # save retrieval
        self.pred_gt_dir = ""
        self.pred_gt_dir_cv = ""

        # save evaluation
        self.eval_result_dir = ""
        self.eval_result_dir_cv = ""

        """------------------ Network Architecture ------------------------------------------------------------------"""
        # self.tf_hub_module = "/bigwork/nhgndoro/tfhub_modules/8d0633058d241900a3e5895be9e610592a5e1fac/"
        self.tf_hub_module = "https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1"
        self.num_fine_tune_layers = -1
        self.add_fc = []

        """------------------ Training Specifications ---------------------------------------------------------------"""
        self.how_many_training_steps = -1
        self.how_often_validation = -1
        self.learning_rate = -1
        # self.weight_decay = -1
        self.optimizer_ind = "Adam"
        self.list_valid_optimizers = ["Adam", "Adagrad", "GradientDescent"]

        self.batch_size = -1
        self.validation_percentage = -1
        self.min_samples_per_class = -1
        self.min_num_labels = -1
        self.image_based_samples = True  # record-based samples (False) or image-based samples (True)?

        """------------------ Similarity Specifications -------------------------------------------------------------"""
        self.relevant_variables = []
        self.variable_weights = list(np.ones(len(self.relevant_variables)))
        self.loss_ind = ""
        self.similarity_thresh = -1
        self.list_valid_losses = ['soft_contrastive_loss', 'soft_triplet_incomp_loss_min_margin',
                                  'soft_triplet_loss', 'hue_saturation_similarity_loss',
                                  'self_augmented_similarity_loss', 'combined_similarity_loss']
        self.colour_distr_resolution = 5

        """------------------ Help Variables for combined similarity ------------------------------------------------"""
        self.all_full_image_names = []
        # weights for weighting the fraction of samples of each dataset contributing to the batch
        # -> 3 different disjoint datasets can be combined
        # TODO: Handle that int number of samples will be drwan from respective dataset
        self.semantic_batch_size_fraction = 1
        self.rules_batch_size_fraction = 0
        self.colour_augment_batch_size_fraction = 0

        # weights for the loss terms in the combined loss
        self.loss_weight_semantic = 1 / 2
        self.loss_weight_rules = 0
        self.loss_weight_colour = 1 / 4
        self.loss_weight_augment = 1 / 4

        """------------------ Retrieval Specifications --------------------------------------------------------------"""
        self.num_neighbors = -1
        self.bool_labeled_input = -1

        """------------------ Augmentation --------------------------------------------------------------------------"""
        self.aug_set_dict = {}

        """------------------ Domain Adaptation ---------------------------------------------------------------------"""
        self.bool_domain_classifier = False
        self.domain_variable = "museum"  # If other variable than "museum": Be careful when using multi labels (logits!!)
        self.factor_sim_loss = 1
        self.factor_domain_loss = 1

    """ -------------------------------- Routines API ---------------------------------------------------------------"""

    def train_model(self):
        """Trains a new siamese network.

            :Arguments\::
                No arguments. All parameters have to be set within the class object.

            :Returns\::
                No returns. The trained graph (containing the tfhub_module and the
                trained classifier) is stored automatically in the directory given in
                the control file.
            """

        # 0. Assertions and Paths
        # =======================
        self.check_valid_inputs()

        # 1. check whether directories exist. If not, create them
        # =======================================================
        if not os.path.exists(os.path.join(self.log_dir, r"")):
            os.makedirs(os.path.join(self.log_dir, r""))
        if not os.path.exists(os.path.join(self.model_dir, r"")):
            os.makedirs(os.path.join(self.model_dir, r""))
        self._copy_collections()

        # 2. collect data samples
        # =======================
        # Initialize the SampleHandler and loads all images
        self.structure_and_load_image_data()

        # pre-compute all hue-saturation matrices in case of colour loss for validation(also in combined loss)
        if self.loss_ind in ["hue_saturation_similarity_loss", "combined_similarity_loss"]:
            self.pre_compute_colour_descriptors()

        # 3. Build Graph
        # =======================
        graph = tf.Graph()
        with graph.as_default():

            # 3.1 integrate pre-processing pipeline
            # =====================================
            # jpeg_data_tensor, in_img_tensor will contain one image, not one batch
            # (scale to input size, potential augmentations, ...)
            # TODO: Impelemnt load_module_spec(path)
            module_spec = hub.load_module_spec(str(self.tf_hub_module))
            (jpeg_data_tensor,
             in_img_tensor) = wp4lib.add_jpeg_decoding(module_spec=module_spec)
            augmented_image_tensor = wp4lib.add_data_augmentation(self.aug_set_dict, in_img_tensor)

            # 3.2 build siamese network...
            # ============================
            # TODO: return user-defined feature vector for domain classifier...
            (in_batch_tensor,
             output_feature_tensor,
             in_batch_tensor_augmented,
             output_feature_tensor_augmented,
             retrain_vars) = self.create_module_graph(module_spec=module_spec,
                                                      reuse=False)

            # ... potentially with domain classifier...
            # TODO: ...and use this user-defined feature vector for domain classifier
            if self.bool_domain_classifier:
                (domain_label_tensor,
                 output_feature_tensor,
                 retrain_vars,
                 domain_loss,
                 bool_reduce_domain) = self.add_domain_classifier(output_feature_tensor,
                                                                  retrain_vars)
            else:
                domain_label_tensor = None
                domain_loss = None
                bool_reduce_domain = None

            # ... with normalized feature vectors...
            output_feature_tensor = tf.nn.l2_normalize(output_feature_tensor,
                                                       axis=-1, name="normalized_descriptor")

            output_feature_tensor_augmented = tf.nn.l2_normalize(output_feature_tensor_augmented,
                                                                 axis=-1, name="normalized_augment_descriptor")

            # ... and similarity loss
            (bool_only_bs_hardest, bool_reduce,
             in_label_tensor, label_weight_tensor,
             norm_cross_corr_plh, similarity_loss,
             rules_indicator_similar, rules_indicator_dissimilar,
             weights_combined_loss_tensor) = self.get_similarity_loss(output_feature_tensor,
                                                                      output_feature_tensor_augmented)

            # total loss
            if self.bool_domain_classifier:
                # TODO: Enable unreduced domainloss for validation??
                # TODO: Weighting of the loss by gradient magnitude
                # "train/DomainClassificationLayer/museum_5_classes/kernel_0/gradient "
                # "train/CustomLayers/output_features/kernel_0/gradient "
                loss = self.factor_sim_loss * tf.reduce_mean(similarity_loss) + \
                       self.factor_domain_loss * domain_loss
            else:
                loss = similarity_loss

            # 3.3 Optimizer
            # =============
            train_step, variable_list = self.setup_optimizer(loss, retrain_vars)

            # 3.4 Count number of trainable parameters
            # ===========================
            self.count_trainable_parameters(variable_list)

            # 3.5 run training
            # ================
            best_validation_loss = None
            train_saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.Session(graph=graph) as sess:
                # 3.5.1: initialize network
                init = tf.compat.v1.global_variables_initializer()
                sess.run(init)

                merged = tf.compat.v1.summary.merge_all()  # TODO: training parameter monitoring
                train_writer = tf.compat.v1.summary.FileWriter(self.log_dir + 'train', sess.graph)
                val_writer = tf.compat.v1.summary.FileWriter(self.log_dir + 'val', sess.graph)

                for train_iter in range(self.how_many_training_steps):
                    # 3.5.2: get batch for training
                    # includes data augmentation
                    (batch_in_img_train,
                     train_ground_truth,
                     train_ground_truth_domain,
                     train_image_name) = self.create_batch(augmented_image_tensor=augmented_image_tensor,
                                                           in_img_tensor=in_img_tensor,
                                                           jpeg_data_tensor=jpeg_data_tensor,
                                                           sess=sess,
                                                           data_creation_purpose="train")

                    # 3.5.3: get rules ground truth for the batch
                    (rules_indicator_similar_values,
                     rules_indicator_dissimilar_values) = self.get_rules_indicator_batch(
                        image_name_batch=train_image_name)

                    # 3.5.4: get colour ground truth for the batch
                    batch_norm_cross_corr = self.get_colour_correlation_batch(train_image_name)

                    # Actual training iteration
                    feed_dict_train = self.create_feed_dict(batch_in_img_train, batch_norm_cross_corr,
                                                            bool_only_bs_hardest,
                                                            bool_reduce, bool_reduce_domain, domain_label_tensor,
                                                            in_batch_tensor, in_batch_tensor_augmented, in_label_tensor,
                                                            label_weight_tensor, norm_cross_corr_plh,
                                                            rules_indicator_dissimilar,
                                                            rules_indicator_dissimilar_values,
                                                            rules_indicator_similar, rules_indicator_similar_values,
                                                            train_ground_truth, train_ground_truth_domain,
                                                            weights_combined_loss_tensor)

                    # ################################### Code development (start) #######################################
                    #
                    # self.output_feature_tensor = output_feature_tensor
                    # self.in_label_tensor = in_label_tensor
                    # squared = False
                    # # S--------------------------------------------------------------------------------------------------
                    # # Get the pairwise distance matrix
                    # # pairwise_dist = self._pairwise_distances(squared=False)
                    #
                    # dot_product = tf.matmul(self.output_feature_tensor, tf.transpose(self.output_feature_tensor))
                    # square_norm = tf.linalg.tensor_diag_part(dot_product)
                    # distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
                    # distances = tf.maximum(distances, 0.0)
                    # if not squared:
                    #     mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
                    #     distances = distances + mask * 1e-16
                    #     distances = tf.sqrt(distances)
                    #     distances = distances * (1.0 - mask)
                    #
                    # pairwise_dist = distances
                    # # E--------------------------------------------------------------------------------------------------
                    #
                    # x1x2_positive_dist = tf.expand_dims(pairwise_dist, 2)
                    # x1x3_negative_dist = tf.expand_dims(pairwise_dist, 1)
                    #
                    # # S-------------------------------------------------------------------------------------------------
                    # # shape(margin) = (batch_size, batch_size, batch_size)
                    # # margin = self._get_incomplete_margin_triplets()
                    #
                    # # S-----------------------------------------------------------------------------
                    # # S-----------------------------------------------------------------------------
                    # # pairwise_similarity, _ = self._get_incomplete_similarity_pairs()
                    #
                    # # label_acc_x1x3 = tf.expand_dims(self.in_label_tensor, 1)
                    # # label_equality_p = tf.math.equal(self.in_label_tensor, label_acc_x1x3)
                    # # label_equality_p = tf.cast(label_equality_p, tf.float32)
                    # # # label_equality_n = tf.math.not_equal(self.in_label_tensor, label_acc_x1x3)
                    # # # label_equality_n = tf.cast(label_equality_n, tf.float32)
                    # # label_nan = tf.cast(tf.math.not_equal(self.in_label_tensor, -1), tf.float32)
                    # # label_similarity_pos = tf.math.reduce_sum(label_equality_p * label_nan, axis=2) / tf.cast(
                    # #     tf.shape(self.in_label_tensor)[1],
                    # #     tf.float32)
                    #
                    # # in_label_tensor_bool_mask = tf.where(self.in_label_tensor == 1, True, False)
                    #
                    # if self.multiLabelsListOfVariables is None:
                    #     print("old Code")
                    #     label_acc_x1x3 = tf.expand_dims(self.in_label_tensor, 1)
                    #
                    #     # shape = (batch_size, batch_size, num_rel_labels)
                    #     label_equality_p = tf.math.equal(self.in_label_tensor, label_acc_x1x3)
                    #     label_equality_p = tf.cast(label_equality_p, tf.float32)
                    #
                    #     label_equality_n = tf.math.not_equal(self.in_label_tensor, label_acc_x1x3)
                    #     label_equality_n = tf.cast(label_equality_n, tf.float32)
                    #
                    #     # shape = (batch_size, num_rel_labels)
                    #     label_nan = tf.cast(tf.math.not_equal(self.in_label_tensor, -1), tf.float32)
                    #
                    # else:
                    #     mask_false = tf.fill(tf.shape(self.in_label_tensor), False)
                    #     mask_true = tf.fill(tf.shape(self.in_label_tensor), True)
                    #     in_label_tensor_bool_mask = tf.where(tf.equal(self.in_label_tensor, 1), mask_true, mask_false)
                    #     label_acc_x1x3_bool_mask = tf.expand_dims(in_label_tensor_bool_mask, 1)
                    #
                    #     label_acc_x1x3 = tf.expand_dims(self.in_label_tensor, 1)
                    #     max_multi_label_divisor = tf.cast(tf.math.maximum(tf.reduce_sum(self.in_label_tensor, axis=2),
                    #                                                       tf.reduce_sum(label_acc_x1x3, axis=3)),
                    #                                       tf.float32)
                    #
                    #     label_equality_p_bool_mask = tf.math.logical_and(in_label_tensor_bool_mask,
                    #                                                      label_acc_x1x3_bool_mask)
                    #     label_equality_p_float_mask = tf.cast(label_equality_p_bool_mask, tf.float32)
                    #     label_equality_p_sum = tf.reduce_sum(label_equality_p_float_mask, axis=3)
                    #     label_equality_p_norm = label_equality_p_sum / (max_multi_label_divisor + 1e-16)
                    #     label_equality_p = label_equality_p_norm
                    #
                    #     label_equality_n_bool_mask = tf.math.logical_not(label_equality_p_bool_mask)
                    #     label_equality_n_float_mask = tf.cast(label_equality_n_bool_mask, tf.float32)
                    #     label_equality_n_sum = tf.reduce_sum(label_equality_n_float_mask, axis=3)
                    #     label_equality_n_norm = label_equality_n_sum / (max_multi_label_divisor + 1e-16)
                    #     label_equality_n = label_equality_n_norm
                    #
                    #     # shape = (batch_size, num_rel_labels)
                    #     label_nan = tf.cast(tf.math.not_equal(tf.reduce_sum(self.in_label_tensor, axis=2), 0),
                    #                         tf.float32)
                    #
                    # # # TODO: insert weights for variables
                    # # # shape = (batch_size, batch_size)
                    # # label_similarity_pos = tf.math.reduce_sum(label_equality_p * label_nan, axis=2) / tf.cast(
                    # #     tf.shape(self.in_label_tensor)[1],
                    # #     tf.float32)
                    #
                    # label_similarity_pos = tf.math.reduce_sum(
                    #     tf.multiply(label_weight_tensor, label_equality_p) * label_nan, axis=2)
                    #
                    # pairwise_similarity = label_similarity_pos
                    # # E-----------------------------------------------------------------------------
                    # # E-----------------------------------------------------------------------------
                    #
                    # Yp_ap = tf.expand_dims(pairwise_similarity, 2)
                    # Yp_an = tf.expand_dims(pairwise_similarity, 1)
                    #
                    # # S-----------------------------------------------------------------------------
                    # # S-----------------------------------------------------------------------------
                    # # pairwise_knowledge = self._get_incomplete_knowledge_pairs()
                    # if self.multiLabelsListOfVariables is None:
                    #     pi_1 = tf.cast(tf.math.not_equal(self.in_label_tensor, -1), tf.float32)
                    # else:
                    #     pi_1 = tf.cast(tf.math.not_equal(tf.reduce_sum(self.in_label_tensor, axis=2), 0), tf.float32)
                    # pi_2 = tf.expand_dims(pi_1, 1)
                    # pi1_times_pi2 = tf.multiply(pi_1, pi_2)
                    # # available_knowledge = tf.math.reduce_sum(pi1_times_pi2, axis=2) / tf.cast(
                    # #     tf.shape(self.in_label_tensor)[1],
                    # #     tf.float32)
                    #
                    # available_knowledge = tf.math.reduce_sum(tf.multiply(self.variable_weights, pi1_times_pi2), axis=2)
                    #
                    # pairwise_knowledge = available_knowledge
                    # # E-----------------------------------------------------------------------------
                    # # E-----------------------------------------------------------------------------
                    #
                    # k_an = tf.expand_dims(pairwise_knowledge, 1)
                    # temp = Yp_an + 1. - k_an
                    # margin_____ = Yp_ap - temp
                    #
                    # margin = margin_____
                    # # E-------------------------------------------------------------------------------------------------
                    #
                    # # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
                    # # triplet_loss[i, j, k] will contain the triplet loss of x1=i, x2=j, x3=k
                    # # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
                    # # and the 2nd (batch_size, 1, batch_size)
                    # triplet_loss = x1x2_positive_dist - x1x3_negative_dist + margin
                    #
                    # # S-------------------------------------------------------------------------------------------------
                    # # Put to zero the invalid triplets
                    # # where i, j, k not distinct
                    # # where margin(anchor, positive, negative) >0,
                    # # i.e. pos.sim(a, p) >= pos.sim(a, n)  + pot.sim(a, n)
                    # # <=>  yp_apn         >= yp_an         + 1-k_an
                    # # mask = self._get_incomplete_mask_triplets()
                    #
                    # indices_equal = tf.cast(tf.eye(tf.shape(self.in_label_tensor)[0]), tf.bool)
                    # indices_not_equal = tf.logical_not(indices_equal)
                    # i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
                    # i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
                    # j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
                    # distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j,
                    #                                                  i_not_equal_k),
                    #                                   j_not_equal_k)
                    #
                    # # S-----------------------------------------------------------------------------
                    # # S-----------------------------------------------------------------------------
                    # # margin = self._get_incomplete_margin_triplets()
                    #
                    # margin = margin  # ggf. prüfen!!!!
                    # # E-----------------------------------------------------------------------------
                    # # E-----------------------------------------------------------------------------
                    #
                    # x = tf.fill(tf.shape(margin), False)
                    # y = tf.fill(tf.shape(margin), True)
                    # mask_similarity = tf.where(margin <= 0, x, y)
                    # mask___ = tf.logical_and(distinct_indices, mask_similarity)
                    #
                    # mask = mask___
                    # # E-------------------------------------------------------------------------------------------------
                    #
                    # mask = tf.cast(mask, tf.float32)
                    # triplet_loss_masked_all = tf.multiply(mask, triplet_loss)
                    #
                    # # Remove negative losses (i.e. the easy triplets)
                    # triplet_loss_masked = tf.maximum(triplet_loss_masked_all, 0.0)
                    #
                    # # Count number of positive triplets (where triplet_loss > 0)
                    # # implicitly removes easy triplets by averaging over all hard and
                    # # semi-hard triplets
                    # # (already correct distance of triplets' features in feature space)
                    # valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
                    # num_positive_triplets = tf.reduce_sum(
                    #     valid_triplets)  # valid triplets in the sense that they are valid and that they are no easy triplets
                    # num_valid_triplets = tf.reduce_sum(mask)  # valid triplets only in the sense that they are valid
                    # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
                    #
                    # # batch hard mining: Focus on hardest pairs for training
                    # # S-----------------------------------------------------------------------------
                    # # S-----------------------------------------------------------------------------
                    # # goal: per anchor the hardest positive and the hardest negative
                    # # hardest positive = min(Y_ap)
                    # #      -----> geringe semantische Ähnlichkeit (bei viel Wissen!?)
                    # # hardest negative = max(Yp_an + 1. - k_an)
                    # #      -----> größte (potentielle) semantische Ähnlichkeit (bei viel Wissen!?)
                    # # insgesamt: je anchor den hardest loss kombiniert wohl beides
                    # test_hardest = tf.reduce_max(tf.reduce_max(triplet_loss_masked, axis=-1), axis=-1)
                    #
                    # # E-----------------------------------------------------------------------------
                    # # E-----------------------------------------------------------------------------
                    #
                    # bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_triplet")
                    # loss_hardest = tf.cond(bool_only_bs_hardest,
                    #                        lambda: tf.reduce_max(tf.reduce_max(triplet_loss_masked, axis=-1), axis=-1),
                    #                        lambda: tf.reshape(triplet_loss_masked,
                    #                                           [tf.shape(triplet_loss_masked)[0] *
                    #                                            tf.shape(triplet_loss_masked)[1] *
                    #                                            tf.shape(triplet_loss_masked)[2]]))
                    #
                    # # Get final mean triplet loss over the positive valid triplets
                    # bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_triplet")
                    # triplet_loss = tf.cond(bool_reduce_mean,
                    #                        lambda: tf.reduce_mean(loss_hardest),
                    #                        lambda: loss_hardest)
                    #
                    # feed_dict[bool_only_bs_hardest] = True
                    # feed_dict[bool_reduce_mean] = True
                    #
                    # (test_hardest_v) = sess.run([test_hardest],
                    #                             feed_dict=feed_dict)
                    #
                    # print("test_hardest_v", test_hardest_v)
                    # print(np.shape(test_hardest_v))
                    #
                    # print(np.mean(test_hardest_v))
                    #
                    #
                    # ################################### Code development (end)   #######################################

                    (cur_loss, _, train_summary) = sess.run([loss, train_step, merged],
                                                            feed_dict=feed_dict_train)
                    print('current train loss (it. ' + str(train_iter) + '):\n', cur_loss)

                    train_writer.add_summary(train_summary, train_iter)
                    train_loss = [tf.compat.v1.Summary.Value(tag='total_loss', simple_value=cur_loss)]
                    train_writer.add_summary(tf.compat.v1.Summary(value=train_loss), train_iter)
                    train_writer.flush()

                    # 3.6 Validation of the current iteration
                    # =======================================
                    if self.validation_percentage > 0 and (train_iter % self.how_often_validation == 0):
                        # get batch for validation
                        # (no data augmentation)
                        (val_image_data,
                         val_ground_truth,
                         val_ground_truth_domain,
                         val_file_names) = self.create_batch(augmented_image_tensor=augmented_image_tensor,
                                                             in_img_tensor=in_img_tensor,
                                                             jpeg_data_tensor=jpeg_data_tensor,
                                                             sess=sess,
                                                             data_creation_purpose="valid")

                        # get rules ground truth for the batch
                        (rules_indicator_similar_values_val,
                         rules_indicator_dissimilar_values_val) = self.get_rules_indicator_batch(
                            image_name_batch=val_file_names)

                        # get colour ground truth for the batch
                        batch_norm_cross_corr_val = self.get_colour_correlation_batch(val_file_names)

                        feed_dict_val = self.create_feed_dict(val_image_data, batch_norm_cross_corr_val,
                                                              bool_only_bs_hardest,
                                                              bool_reduce, bool_reduce_domain, domain_label_tensor,
                                                              in_batch_tensor, in_batch_tensor_augmented,
                                                              in_label_tensor,
                                                              label_weight_tensor, norm_cross_corr_plh,
                                                              rules_indicator_dissimilar,
                                                              rules_indicator_dissimilar_values_val,
                                                              rules_indicator_similar,
                                                              rules_indicator_similar_values_val,
                                                              val_ground_truth, val_ground_truth_domain,
                                                              weights_combined_loss_tensor)

                        (val_loss,
                         val_loss_sim) = sess.run([loss, similarity_loss], feed_dict=feed_dict_val)

                        val_loss_tf = [tf.compat.v1.Summary.Value(tag='total_loss', simple_value=val_loss)]
                        val_writer.add_summary(tf.compat.v1.Summary(value=val_loss_tf), train_iter)
                        val_loss_sim_tf = [tf.compat.v1.Summary.Value(tag=self.loss_ind, simple_value=val_loss_sim)]
                        val_writer.add_summary(tf.compat.v1.Summary(value=val_loss_sim_tf), train_iter)
                        val_writer.flush()

                        print('current val loss (it. ' + str(train_iter) + '):\n', val_loss_sim)

                        # Save (best) configuration based on similarity loss
                        # -> domain loss not considered if self.bool_domain_classifier
                        # TODO: filter last few iterations to avoid "noise peaks"
                        # TODO: Write the corresponding training iteration to a file
                        if best_validation_loss is None or val_loss_sim < best_validation_loss:
                            print("New best model found!")
                            train_saver.save(sess, self.model_dir + '/' + 'model.ckpt')
                            best_validation_loss = val_loss_sim

        self._write_train_parameters_to_configuration_file()

    def check_valid_inputs(self):
        assert np.sum(self.variable_weights)== 1., "The sum of weights for the variables has to be one."
        assert 0 <= self.validation_percentage < 100, "Validation Percentage has to be between 0 and 100!"
        assert self.loss_ind in self.list_valid_losses, "This loss is not implemented yet or may the key is written\
                      incorrectly."
        assert self.optimizer_ind in self.list_valid_optimizers, "This optimizer is not implemented yet or" \
                                                                 "may the key is written incorrectly."
        assert (self.loss_weight_semantic + self.loss_weight_rules + self.loss_weight_colour +
                self.loss_weight_augment) == 1.0, "The sum of weights for the loss term is not one."
        assert (self.semantic_batch_size_fraction + self.rules_batch_size_fraction +
                self.colour_augment_batch_size_fraction) == 1.0, "The sum of weights for the batch creation is not one."
        if self.master_file_rules_name == "" and self.loss_ind == "combined_similarity_loss" \
                and self.rules_batch_size_fraction > 0:
            print("master file for rules needs to be defined")
            sys.exit()
        # if self.master_file_colour_augment_name == "" and self.loss_ind == "combined_similarity_loss":
        #     print("master file for colour augment needs to be defined")
        #     sys.exit()

    def build_kd_tree(self):
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
        # Checks for paths existing: model_path, master file, savepath
        if not os.path.exists(os.path.join(self.tree_dir, r"")):
            os.makedirs(os.path.join(self.tree_dir, r""))

        # Load pre-trained network
        model = ImportGraph(self.model_dir)

        # load samples from which the tree will be created
        coll_list = wp4lib.master_file_to_collections_list(self.master_dir_tree, self.master_file_tree)
        coll_dict, data_dict = wp4lib.collections_list_MTL_to_image_lists(collections_list=coll_list,
                                                                          labels_2_learn=self.relevant_variables,
                                                                          master_dir=self.master_dir_tree,
                                                                          multiLabelsListOfVariables=self.multiLabelsListOfVariables)
        if self.loss_ind in ["combined_similarity_loss", "prior_knowledge_similarity_loss"] \
                and self.rules_batch_size_fraction > 0:
            coll_list_rule = wp4lib.master_file_to_collections_list(self.master_dir_tree, self.master_file_rules_name)
            _, data_dict_rule = wp4lib.collections_list_MTL_to_image_lists(
                collections_list=coll_list_rule,
                labels_2_learn=self.relevant_variables,
                master_dir=self.master_dir_tree,
                bool_unlabeled_dataset=True)
            data_dict = {**data_dict, **data_dict_rule}


        cur_dir = os.path.abspath(os.getcwd())
        rel_paths = [os.path.relpath(i, cur_dir) for i in data_dict.keys()]
        tree_data_dict = data_dict
        for (old_key, new_key) in zip(list(data_dict.keys()), rel_paths):
            tree_data_dict[new_key] = tree_data_dict.pop(old_key)

        # lists begin with task name, followed by all corresponding classes
        label2class_list = []
        for label_key in self.relevant_variables:
            var_name = np.asarray(label_key)
            if self.multiLabelsListOfVariables is None:
                var_list = np.asarray(list(coll_dict[label_key].keys()))
            else:
                var_list = []
                for cl in list(coll_dict[label_key].keys()):
                    if "___" not in cl:
                        var_list.append(cl)
                    else:
                        for singleLabel in cl.split("___"):
                            var_list.append(singleLabel)
                var_list = np.asarray(np.unique(var_list))
            var_list = np.insert(var_list, 0, var_name)
            label2class_list.append(var_list)

        # get feature vectors and labels, for all samples
        features_all = []
        labels_all = []
        print("Estimating descriptors:")
        for image_name in tqdm(data_dict.keys()):

            # Get labels, check for incompleteness
            labels_var = []
            for variable in self.relevant_variables:
                labels_var.append(data_dict[image_name][variable])

            # Get feature vector
            image_data = tf.io.gfile.GFile(image_name, 'rb').read()
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
                            "relevant_variables": self.relevant_variables,
                            "label2class_list": label2class_list
                            }
        np.savez(self.tree_dir + r"/kdtree.npz", tree_with_labels)

    def get_knn(self):
        if not os.path.exists(os.path.join(self.pred_gt_dir, r"")):
            os.makedirs(os.path.join(self.pred_gt_dir, r""))

        # Load pre-trained network
        model = ImportGraph(self.model_dir)

        # Load Tree and labels
        (tree, labels_tree, data_dict_train, relevant_variables, label2class_list) = self.load_kd_tree()

        # Load query images
        if self.bool_labeled_input:
            image_list, labels_target = self._get_query_image_list(relevant_variables=relevant_variables)
        else:
            image_list = self._get_query_image_list(relevant_variables=relevant_variables)
            labels_target = []

        # Calculate descriptors
        features = self._get_descriptors(model=model, image_list=image_list)

        # Perform image retrieval
        dist, ind = tree.query(np.squeeze(features), k=self.num_neighbors)

        # Get labels of knn (pred_top_k) and knn-classification (actual prediction)
        (pred_label_test,
         pred_names_test,
         pred_occ_test,
         pred_top_k) = self._get_knn_labels(labels_tree=labels_tree,
                                            tree_index_acc_labels=ind,
                                            relevant_variables=relevant_variables,
                                            label2class_list=label2class_list)

        # Save results
        self._write_knn_list_and_lut(query_image_list=image_list, tree_image_list=list(data_dict_train.keys()),
                                     labels_target=labels_target, pred_names_test=pred_names_test,
                                     labels_tree=labels_tree, tree_indices_knn=ind,
                                     tree_dist_knn=dist, relevant_variables=relevant_variables)
        if self.bool_labeled_input:
            pred_gt = {"Groundtruth": np.asarray(labels_target),  # indices der Klassen 0, 1, ...
                       "Predictions": np.asarray(pred_names_test),  # indices der Klassen 0, 1, ...
                       "label2class_list": np.asarray(label2class_list)}  # konkrete Namen
            np.savez(self.pred_gt_dir + r"/pred_gt.npz", pred_gt)

            # TODO: raus für D4.5
            # TODO: D4.6: Aufschlüsseln nach Variable und Klasse
            # TODO: D4.6: neuerliche Evaluierung raus, wird ja zu Standardeval
            # For the estimation of a suitable k
            # get_top_k_statistics(pred_top_k,
            #                      labels_target,
            #                      num_neighbors,
            #                      savepath,
            #                      label2class_list)

        return pred_top_k, labels_target, label2class_list

    def evaluate_model(self):
        if not os.path.exists(os.path.join(self.eval_result_dir, r"")):
            os.makedirs(os.path.join(self.eval_result_dir, r""))

        # Load predictions and ground truth
        var_dict = np.load(self.pred_gt_dir + r"/pred_gt.npz", allow_pickle=True)["arr_0"].item()
        all_var_predictions = np.asarray(var_dict["Predictions"])
        all_var_ground_truth = np.asarray(var_dict["Groundtruth"])
        label2class_list = np.asarray(var_dict["label2class_list"])

        for task_ind, class_list in enumerate(label2class_list):
            task_name = class_list[0]
            list_class_names = class_list[1:]

            # sort out nans
            gt_var = all_var_ground_truth[:, task_ind]
            pr_var = all_var_predictions[:, task_ind]

            nan_mask_gt = gt_var != 'nan'
            nan_mask_pr = pr_var != 'nan'
            nan_mask = np.logical_and(nan_mask_gt, nan_mask_pr)

            gt_var = gt_var[nan_mask]
            pr_var = pr_var[nan_mask]

            # identify underrepresented tasks
            gt_unique, gt_counts = np.unique(gt_var, return_counts=True)
            if len(gt_unique) < len(list_class_names):
                print('WARNING: The variable {0!r} has no contribution for '
                      'at least one class in the ground truth. The variable will not be evaluated!'.format(task_name))
                continue
            if any(gt_counts) < 20:
                print('WARNING: The variable {0!r} has a contribution of less than 20 samples for '
                      'at least one class in the ground truth. The evaluation will probably not be '
                      'representative!'.format(task_name))

            wp4lib.estimate_multi_label_quality_measures(gtvar=gt_var,
                                                         prvar=pr_var,
                                                         list_class_names=list_class_names,
                                                         result_dir=self.eval_result_dir,
                                                         multiLabelsListOfVariables=self.multiLabelsListOfVariables,
                                                         taskname = task_name)

        return all_var_ground_truth, all_var_predictions

    def cross_validation(self):
        # load collections
        coll_list = wp4lib.master_file_to_collections_list(self.master_file_dir, self.master_file_name_cv)
        if self.loss_ind in ["combined_similarity_loss", "prior_knowledge_similarity_loss"] \
                and self.rules_batch_size_fraction > 0:
            coll_list_rules = wp4lib.master_file_to_collections_list(self.master_file_dir,
                                                                     self.master_file_rules_name_cv)

        # for collecting all predictions and ground truth among the cross val iter
        all_predictions = []
        all_ground_truth = []

        # FIVE cross validation iterations
        self.master_dir_tree = self.master_file_dir
        self.master_dir_retrieval = self.master_file_dir
        self.bool_labeled_input = True

        for cv_iter in range(5):
            # create intermediate master files for sub-modules (semantic loss)
            if str(self.log_dir_cv)[-1] == "/":
                temp_exp_name = str(self.log_dir_cv).split("/")[-3]
            else:
                temp_exp_name = str(self.log_dir_cv).split("/")[-2]
            temp_train_master_name = 'train_master_' + temp_exp_name + '.txt'
            temp_test_master_name = 'test_master_' + temp_exp_name + '.txt'
            train_master = open(os.path.abspath(os.path.join(self.master_file_dir, temp_train_master_name)), 'w')
            test_master = open(os.path.abspath(os.path.join(self.master_file_dir, temp_test_master_name)), 'w')
            train_coll = np.roll(coll_list, cv_iter)[:-1]
            test_coll = np.roll(coll_list, cv_iter)[-1]
            for c in train_coll:
                train_master.write("%s\n" % c)
            test_master.write(test_coll)
            train_master.close()
            test_master.close()

            # create intermediate master files for sub-modules (rules loss)
            if self.loss_ind in ["combined_similarity_loss", "prior_knowledge_similarity_loss"] \
                    and self.rules_batch_size_fraction > 0:
                train_coll_rules = np.roll(coll_list_rules, cv_iter)[:-1]
                test_coll_rules = np.roll(coll_list_rules, cv_iter)[-1]
                temp_train_master_name_rules = 'train_master_' + temp_exp_name + '_rules.txt'
                temp_test_master_name_rules = 'test_master_' + temp_exp_name + '_rules.txt'
                train_master_rules = open(
                    os.path.abspath(os.path.join(self.master_file_dir, temp_train_master_name_rules)), 'w')
                test_master_rules = open(
                    os.path.abspath(os.path.join(self.master_file_dir, temp_test_master_name_rules)), 'w')
                for c in train_coll_rules:
                    train_master_rules.write("%s\n" % c)
                test_master_rules.write(test_coll_rules)
                train_master_rules.close()
                test_master_rules.close()

                self.master_file_rules_name = temp_train_master_name_rules
                self.master_file_retrieval_rules = temp_test_master_name_rules
                self.bool_in_cv=True

            """---------------------- perform training --------------------------------------------------------------"""
            self.master_file_name = temp_train_master_name

            log_dir_cur_cv = self.log_dir_cv + r"/cv" + str(cv_iter) + "/"
            if not os.path.exists(log_dir_cur_cv):
                os.makedirs(log_dir_cur_cv)
            self.log_dir = log_dir_cur_cv

            model_dir_cur_cv = self.model_dir_cv + r"/cv" + str(cv_iter) + "/"
            if not os.path.exists(model_dir_cur_cv):
                os.makedirs(model_dir_cur_cv)
            self.model_dir = model_dir_cur_cv
            self.train_model()

            """---------------------- build tree --------------------------------------------------------------------"""
            self.master_file_tree = temp_train_master_name
            tree_dir_cur_cv = self.tree_dir_cv + r"/cv" + str(cv_iter) + "/"
            if not os.path.exists(tree_dir_cur_cv):
                os.makedirs(tree_dir_cur_cv)
            self.tree_dir = tree_dir_cur_cv
            self.build_kd_tree()

            """---------------------- retrieval ---------------------------------------------------------------------"""
            self.master_file_retrieval = temp_test_master_name
            pred_gt_dir_cur_cv = self.pred_gt_dir_cv + r"/cv" + str(cv_iter) + "/"
            if not os.path.exists(pred_gt_dir_cur_cv):
                os.makedirs(pred_gt_dir_cur_cv)
            self.pred_gt_dir = pred_gt_dir_cur_cv
            self.get_knn()

            """---------------------- evaluation --------------------------------------------------------------"""
            eval_result_dir_cur_cv = self.eval_result_dir_cv + r"/cv" + str(cv_iter) + "/"
            if not os.path.exists(eval_result_dir_cur_cv):
                os.makedirs(eval_result_dir_cur_cv)
            self.eval_result_dir = eval_result_dir_cur_cv
            cur_ground_truth, cur_predictions = self.evaluate_model()

            # concatenate predictions and ground truths
            if len(all_predictions) == 0:
                all_predictions = cur_predictions
                all_ground_truth = cur_ground_truth
            else:
                all_predictions = np.concatenate((all_predictions, cur_predictions))
                all_ground_truth = np.concatenate((all_ground_truth, cur_ground_truth))

            # delete intermediate data
            if os.path.isfile(os.path.join(self.master_file_dir, temp_train_master_name)):
                os.remove(os.path.join(self.master_file_dir, temp_train_master_name))
            if os.path.isfile(os.path.join(self.master_file_dir, temp_test_master_name)):
                os.remove(os.path.join(self.master_file_dir, temp_test_master_name))
            if self.rules_batch_size_fraction > 0:
                if os.path.isfile(os.path.join(self.master_file_dir, temp_train_master_name_rules)):
                    os.remove(os.path.join(self.master_file_dir, temp_train_master_name_rules))
                if os.path.isfile(os.path.join(self.master_file_dir, temp_test_master_name_rules)):
                    os.remove(os.path.join(self.master_file_dir, temp_test_master_name_rules))

        # estimate quality measures with all predictions and ground truths
        var_dict = np.load(pred_gt_dir_cur_cv + r"/pred_gt.npz", allow_pickle=True)["arr_0"].item()
        label2class_list = np.asarray(var_dict["label2class_list"])

        for task_ind, class_list in enumerate(label2class_list):
            task_name = class_list[0]
            list_class_names = np.asarray(class_list[1:])

            # sort out nans
            cur_ground_truth = all_ground_truth[:, task_ind]
            cur_predictions = all_predictions[:, task_ind]
            nan_mask = cur_ground_truth != 'nan'
            nan_mask_pr = cur_predictions != 'nan'
            nan_mask = np.logical_and(nan_mask, nan_mask_pr)

            cur_ground_truth = cur_ground_truth[nan_mask]
            cur_predictions = cur_predictions[nan_mask]

            wp4lib.estimate_multi_label_quality_measures(gtvar=cur_ground_truth,
                                                         prvar=cur_predictions,
                                                         list_class_names=list_class_names,
                                                         result_dir=self.eval_result_dir_cv,
                                                         multiLabelsListOfVariables=self.multiLabelsListOfVariables,
                                                         taskname=task_name)

            # ground_truth = np.squeeze([np.where(gt == list_class_names) for gt in cur_ground_truth])
            # prediction = np.squeeze([np.where(pr == list_class_names) for pr in cur_predictions])
            # wp4lib.estimate_quality_measures(ground_truth=ground_truth,
            #                                  prediction=prediction,
            #                                  list_class_names=list(list_class_names),
            #                                  prefix_plot=task_name,
            #                                  res_folder_name=self.eval_result_dir_cv)

    """ -------------------------------- Routines CNN ---------------------------------------------------------------"""

    def create_module_graph(self, module_spec, reuse):
        """**************MODULE GRAPH********************"""
        height, width = hub.get_expected_image_size(module_spec)
        with tf.compat.v1.variable_scope("ModuleLayers"):
            in_batch_tensor = tf.compat.v1.placeholder(tf.float32, [None, height, width, 3], name="input_img")
            in_batch_tensor_augmented = tf.compat.v1.placeholder(tf.float32, [None, height, width, 3],
                                                                 name="input_img_augmented")
            tensor_augmented = tf.map_fn(
                lambda img_tensor: wp4lib.add_data_augmentation(self.aug_set_dict, img_tensor),
                in_batch_tensor_augmented)

        if self.num_fine_tune_layers > 0 and not reuse:
            if True:  # 'resnet' in self.tf_hub_module:
                # feature computation
                module = hub.Module(module_spec, trainable=True)
                output_module = module(in_batch_tensor)
                output_module_augmented = module(tensor_augmented)
                # get variables to retrain
                help_string_1 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                            scope='module/resnet')[0].name
                help_string_2 = help_string_1.split('/')[0] + '/' + help_string_1.split('/')[1] + '/block'

                temp_choice = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
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
                        if num_added_res_blocks < self.num_fine_tune_layers:
                            retrain_vars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                            scope=help_string_2 + str(
                                                                                makro_block) + '/unit_'
                                                                                  + str(res_block)))
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
            output_module_augmented = m(tensor_augmented)
        """**************\MODULE GRAPH********************"""

        """**************CUSTOM GRAPH********************"""
        init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        with tf.compat.v1.variable_scope('CustomLayers'):
            if len(self.add_fc) == 1:
                output_feature_tensor = tf.layers.dense(inputs=output_module,
                                                        units=self.add_fc[-1],
                                                        use_bias=True,
                                                        kernel_initializer=init,
                                                        activation=None,
                                                        name='output_features',
                                                        reuse=reuse)
                # TODO: debug
                output_feature_tensor_augmented = tf.layers.dense(inputs=output_module_augmented,
                                                                  units=self.add_fc[-1],
                                                                  use_bias=True,
                                                                  kernel_initializer=init,
                                                                  activation=None,
                                                                  name='output_features',
                                                                  reuse=True)
            elif len(self.add_fc) > 1:
                for cur_fc in range(len(self.add_fc) - 1):
                    if cur_fc == 0:
                        dense_layer = tf.layers.dense(inputs=output_module,
                                                      units=self.add_fc[cur_fc],
                                                      use_bias=True,
                                                      kernel_initializer=init,
                                                      activation=tf.nn.relu,
                                                      name='fc_layer' + str(cur_fc) +
                                                           '_' + str(self.add_fc[cur_fc]),
                                                      reuse=reuse)
                        dense_layer_augmented = tf.layers.dense(inputs=output_module_augmented,
                                                                units=self.add_fc[cur_fc],
                                                                use_bias=True,
                                                                kernel_initializer=init,
                                                                activation=tf.nn.relu,
                                                                name='fc_layer' + str(cur_fc) +
                                                                     '_' + str(self.add_fc[cur_fc]),
                                                                reuse=True)
                    else:
                        dense_layer = tf.layers.dense(inputs=dense_layer,
                                                      units=self.add_fc[cur_fc],
                                                      use_bias=True,
                                                      kernel_initializer=init,
                                                      activation=tf.nn.relu,
                                                      name='fc_layer' + str(cur_fc) +
                                                           '_' + str(self.add_fc[cur_fc]),
                                                      reuse=reuse)
                        dense_layer_augmented = tf.layers.dense(inputs=dense_layer_augmented,
                                                                units=self.add_fc[cur_fc],
                                                                use_bias=True,
                                                                kernel_initializer=init,
                                                                activation=tf.nn.relu,
                                                                name='fc_layer' + str(cur_fc) +
                                                                     '_' + str(self.add_fc[cur_fc]),
                                                                reuse=True)
                output_feature_tensor = tf.layers.dense(inputs=dense_layer,
                                                        units=self.add_fc[-1],
                                                        use_bias=True,
                                                        kernel_initializer=init,
                                                        activation=None,
                                                        name='output_features',
                                                        reuse=reuse)
                output_feature_tensor_augmented = tf.layers.dense(inputs=dense_layer_augmented,
                                                                  units=self.add_fc[-1],
                                                                  use_bias=True,
                                                                  kernel_initializer=init,
                                                                  activation=None,
                                                                  name='output_features',
                                                                  reuse=True)
            else:
                output_feature_tensor = output_module
                output_feature_tensor_augmented = output_module_augmented
        """**************\CUSTOM GRAPH********************"""

        return in_batch_tensor, output_feature_tensor, in_batch_tensor_augmented, \
               output_feature_tensor_augmented, retrain_vars

    def add_domain_classifier(self, input_tensor, retrain_vars):
        with tf.compat.v1.variable_scope('GradientReversalLayer'):
            forward_path = tf.stop_gradient(input_tensor * tf.cast(2., tf.float32))
            backward_path = -input_tensor * tf.cast(1., tf.float32)
            output_feature_tensor = forward_path + backward_path

        retrain_vars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                        scope='GradientReversalLayer'))

        init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        with tf.compat.v1.variable_scope('DomainClassificationLayer'):
            logits = tf.layers.dense(inputs=output_feature_tensor,
                                     units=len(self.sample_handler.classCountDict[self.domain_variable]),
                                     use_bias=True,
                                     kernel_initializer=init,
                                     activation=None,
                                     name=str(
                                         self.domain_variable) + '_' + str(
                                         len(self.sample_handler.classCountDict[self.domain_variable])) + '_classes')
            domain_label_tensor = tf.compat.v1.placeholder(tf.int32, [None], name="domain_label_tensor")
            domain_labels_one_hot = tf.one_hot(domain_label_tensor,
                                               depth=int(len(self.sample_handler.classCountDict[self.domain_variable])))

            softmax_cross_entropy_loss_prelim = tf.compat.v1.losses.softmax_cross_entropy(
                onehot_labels=domain_labels_one_hot,
                logits=logits,
                reduction=tf.losses.Reduction.NONE)
            # TODO: Think about clipping by value/gradient/weighing of losses
            # softmax_cross_entropy_loss_clip = tf.clip_by_value(softmax_cross_entropy_loss_prelim,
            #                                               clip_value_min = -self.factor_sim_loss,
            #                                               clip_value_max=self.factor_sim_loss)
            softmax_cross_entropy_loss_clip = softmax_cross_entropy_loss_prelim
            bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean")
            softmax_cross_entropy_loss = tf.cond(bool_reduce_mean,
                                                 lambda: tf.reduce_mean(softmax_cross_entropy_loss_clip),
                                                 lambda: softmax_cross_entropy_loss_clip)

            tf.compat.v1.summary.scalar("domain_loss", softmax_cross_entropy_loss)
            print('A domain classification loss with gradient reversal layer will be maximized.')
            # domain_softmax = tf.nn.softmax(logits,
            #                                name='domain_classifier_' + str(self.domain_variable))
        retrain_vars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                        scope='DomainClassificationLayer'))

        return domain_label_tensor, output_feature_tensor, retrain_vars, softmax_cross_entropy_loss, bool_reduce_mean

    def create_feed_dict(self, batch_in_img_train, batch_norm_cross_corr, bool_only_bs_hardest, bool_reduce,
                         bool_reduce_domain, domain_label_tensor, in_batch_tensor, in_batch_tensor_augmented,
                         in_label_tensor, label_weight_tensor, norm_cross_corr_plh, rules_indicator_dissimilar,
                         rules_indicator_dissimilar_values, rules_indicator_similar, rules_indicator_similar_values,
                         train_ground_truth, train_ground_truth_domain, weights_combined_loss_tensor):
        # print("batch_in_img_train", np.shape(batch_in_img_train))
        # print("rules_indicator_dissimilar_values", np.shape(rules_indicator_dissimilar_values))
        # print("batch_norm_cross_corr", np.shape(batch_norm_cross_corr))
        if self.multiLabelsListOfVariables is None:
            train_ground_truth = np.transpose(train_ground_truth)
        feed_dict = {in_batch_tensor: batch_in_img_train,
                     in_label_tensor: train_ground_truth,
                     label_weight_tensor: self.variable_weights}
        if self.loss_ind not in ['combined_similarity_loss']:
            feed_dict[bool_only_bs_hardest] = True
            feed_dict[bool_reduce] = True
        if self.loss_ind in ['self_augmented_similarity_loss']:
            # already online augmented images (=train_image_data) will be further augmented
            # may feed original images (= train_image_data) to be augmented with other random parameters
            feed_dict[in_batch_tensor_augmented] = batch_in_img_train
        if self.loss_ind in ['hue_saturation_similarity_loss']:
            feed_dict[norm_cross_corr_plh] = batch_norm_cross_corr
        if self.bool_domain_classifier:
            feed_dict[domain_label_tensor] = np.transpose(train_ground_truth_domain)
            feed_dict[bool_reduce_domain] = True
        if self.loss_ind in ['combined_similarity_loss']:
            # already online augmented images (=train_image_data) will be further augmented
            # may feed original images (= train_image_data) to be augmented with other random parameters
            feed_dict[in_batch_tensor_augmented] = batch_in_img_train
            feed_dict[norm_cross_corr_plh] = batch_norm_cross_corr

            # feed rules indicator
            # -> is empty in case of loss_weight_rules == 0
            # -> thus, rules loss = 0
            feed_dict[rules_indicator_similar] = rules_indicator_similar_values
            feed_dict[rules_indicator_dissimilar] = rules_indicator_dissimilar_values

            # the 4 components of the loss will be reduced before combining
            # the combined loss just weights the reduced (averaged) loss terms
            bool_reduce_values = [True, True, True, True, True]
            for b_val in range(np.shape(bool_reduce_values)[0]):
                feed_dict[bool_reduce[b_val]] = bool_reduce_values[b_val]

            # if True, the batch_size hardest losses will be returned
            # self augment and thus, combined, too, have already (always!) len = batch_size
            bool_only_bs_hardest_values = [False, False, False]
            for b_val in range(np.shape(bool_only_bs_hardest_values)[0]):
                feed_dict[bool_only_bs_hardest[b_val]] = bool_only_bs_hardest_values[b_val]

            weights_combined_loss_values = [self.loss_weight_semantic,
                                            self.loss_weight_rules,
                                            self.loss_weight_colour,
                                            self.loss_weight_augment]
            feed_dict[weights_combined_loss_tensor] = weights_combined_loss_values
        return feed_dict

    def get_similarity_loss(self, output_feature_tensor, output_feature_tensor_augmented):
        if self.multiLabelsListOfVariables is None:
            in_label_tensor = tf.compat.v1.placeholder(tf.float16,
                                                       [None, len(self.relevant_variables)],
                                                       name="in_label_tensor")
        else:
            in_label_tensor = tf.compat.v1.placeholder(tf.float16,
                                                       [None, len(self.relevant_variables), None],
                                                       name="in_label_tensor")
        # not used by now
        label_weight_tensor = tf.compat.v1.placeholder(tf.float32,
                                                       [len(self.relevant_variables)],
                                                       name="label_weight_tensor")
        # only for colour similarity
        norm_cross_corr_plh = tf.compat.v1.placeholder(tf.float32,
                                                       [None, None], name="normalized_cross_correlation")
        # only used for rules loss
        rules_indicator_similar = tf.compat.v1.placeholder(tf.float32,
                                                           [None, None], name="rules_indicator_similar")
        rules_indicator_dissimilar = tf.compat.v1.placeholder(tf.float32,
                                                              [None, None], name="rules_indicator_dissimilar")
        weights_combined_loss_tensor = tf.compat.v1.placeholder(tf.float32,
                                                                [4], name="loss_term_weight_tensor")
        loss_instance = SimilarityLosses(relevant_variables=self.relevant_variables,
                                         in_label_tensor=in_label_tensor,
                                         output_feature_tensor=output_feature_tensor,
                                         label_weight_tensor=label_weight_tensor,
                                         similarity_thresh=self.similarity_thresh,
                                         loss_ind=self.loss_ind,
                                         batch_size=self.batch_size,
                                         output_feature_tensor_augmented=output_feature_tensor_augmented,
                                         rules_indicator_similar=rules_indicator_similar,
                                         rules_indicator_dissimilar=rules_indicator_dissimilar,
                                         norm_cross_corr_plh=norm_cross_corr_plh,
                                         weights_combined_loss=weights_combined_loss_tensor,
                                         multiLabelsListOfVariables=self.multiLabelsListOfVariables)
        similarity_loss, bool_reduce, bool_only_bs_hardest = loss_instance.setup_loss()
        return (bool_only_bs_hardest, bool_reduce, in_label_tensor, label_weight_tensor,
                norm_cross_corr_plh, similarity_loss, rules_indicator_similar, rules_indicator_dissimilar,
                weights_combined_loss_tensor)

    def count_trainable_parameters(self, variable_list):
        total_parameters = 0
        for variable in variable_list:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
                total_parameters += variable_parameters
        print('Total Number of parameters:', total_parameters)

    def setup_optimizer(self, loss, retrain_vars):
        # TODO: own learning rate and thus own optimizer for domain_loss?
        with tf.name_scope("train"):
            if self.optimizer_ind == 'Adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer_ind == 'Adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif self.optimizer_ind == 'GradientDescent':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            if self.num_fine_tune_layers == 0:
                grad_var_list = optimizer.compute_gradients(loss,
                                                            tf.compat.v1.trainable_variables())
                # for (grad, var) in grad_var_list:
                #     tf.compat.v1.summary.histogram(var.name.replace(':', '_') + '/gradient',
                #                                    grad)
                #     tf.compat.v1.summary.histogram(var.op.name, var)

                train_step = optimizer.apply_gradients(grad_var_list)
                variable_list = tf.compat.v1.trainable_variables()
            else:
                retrain_vars.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                                scope='CustomLayers'))
                grad_var_list = optimizer.compute_gradients(loss,
                                                            retrain_vars)

                train_step = optimizer.apply_gradients(grad_var_list)
                # train_step = optimizer.minimize(loss, var_list=retrain_vars)
                variable_list = retrain_vars
        return train_step, variable_list

    """ -------------------------------- Routines kNN ---------------------------------------------------------------"""

    def load_kd_tree(self):
        tree = np.load(self.tree_dir + r"/kdtree.npz", allow_pickle=True)["arr_0"].item()
        labels_tree = np.squeeze(tree["Labels"])
        data_dict_train = tree["DictTrain"]
        relevant_variables = tree["relevant_variables"]
        label2class_list = tree["label2class_list"]
        tree = tree["Tree"]

        return tree, labels_tree, data_dict_train, relevant_variables, label2class_list

    def _get_query_image_list(self, relevant_variables):
        coll_list = wp4lib.master_file_to_collections_list(self.master_dir_retrieval, self.master_file_retrieval)
        if self.loss_ind in ["combined_similarity_loss", "prior_knowledge_similarity_loss"] \
                and self.rules_batch_size_fraction > 0 and self.bool_in_cv:
            coll_list_rules = wp4lib.master_file_to_collections_list(self.master_dir_retrieval,
                                                                     self.master_file_retrieval_rules)
            coll_list = np.hstack((coll_list, coll_list_rules))

        if self.bool_labeled_input:
            coll_dict, data_dict = wp4lib.collections_list_MTL_to_image_lists(
                                            collections_list=coll_list,
                                            labels_2_learn=relevant_variables,
                                            master_dir=self.master_dir_retrieval,
                                            bool_unlabeled_dataset=True,
                                            multiLabelsListOfVariables=self.multiLabelsListOfVariables)

            image_list = list(data_dict.keys())
            # Get labels, check for incompleteness
            labels_target = []
            for image_name in image_list:
                labels_var = []
                for variable in relevant_variables:
                    labels_var.append(np.squeeze(data_dict[image_name][variable]))
                labels_target.append(labels_var)

            return image_list, labels_target
        else:
            image_list = []
            for collection in coll_list:
                coll_id = open(os.path.join(self.master_dir_retrieval, collection).strip("\n"), 'r')
                for line, rel_im_path in enumerate(coll_id):
                    if line == 0:
                        continue
                    image_name = os.path.abspath(os.path.join(self.master_dir_retrieval,
                                                              rel_im_path.split('\t')[0].strip("\n")))
                    image_list.append(os.path.relpath(image_name))
                coll_id.close()

            return image_list

    @staticmethod
    def _get_descriptors(model, image_list):
        features = []
        for image_name in image_list:
            image_data = tf.io.gfile.GFile(image_name, 'rb').read()
            features_var = model.run(image_data)
            features.append(features_var)

        return features

    def _get_knn_labels(self, labels_tree, tree_index_acc_labels, relevant_variables, label2class_list):
        pred_label_test = []
        pred_names_test = []
        pred_occ_test = []
        pred_top_k = []
        if len(relevant_variables) == 1:
            task_name=relevant_variables[0]
            for k_neighbors in range(np.shape(tree_index_acc_labels)[0]):
                # list of class labels for all num_neighbors nearest neighbors
                # of the feature vector number k_neighbors in the test set
                temp_pred_list = list(itemgetter(*tree_index_acc_labels[k_neighbors])(labels_tree))

                # majority vote without nan-predictions (nan-pred only if all NN nan)
                (temp_pred_label,
                 temp_pred_name,
                 temp_pred_occ) = self.kNN_classification(task_predictions=np.asarray(temp_pred_list),
                                                          task_name=task_name,
                                                          label2class_list=label2class_list)

                pred_label_test.append(temp_pred_label)
                pred_names_test.append(temp_pred_name)
                pred_occ_test.append(temp_pred_occ)
                pred_top_k.append(np.asarray(temp_pred_list))
            pred_names_test = np.expand_dims(pred_names_test, axis=1)
        elif len(relevant_variables) > 1:
            for k_neighbors in range(np.shape(tree_index_acc_labels)[0]):
                # list of class labels for all num_neighbors nearest neighbors
                # in the train set for the feature vector number k_neighbors
                # in the test set
                temp_pred_list = list(itemgetter(*tree_index_acc_labels[k_neighbors])(labels_tree))

                # majority vote without nan-predictions (nan-pred only if all NN nan)
                temp_pred_label = []
                temp_pred_name = []
                temp_pred_occ = []
                for task_ind in range(len(temp_pred_list[0])):
                    task_predictions = np.asarray(temp_pred_list)[:, task_ind]
                    task_name = relevant_variables[task_ind]
                    class_list = np.asarray(label2class_list)[task_ind]
                    (task_pred_label,
                     task_pred_name,
                     task_pred_occ) = self.kNN_classification(task_predictions=task_predictions,
                                                              task_name=task_name,
                                                              label2class_list=class_list)
                    temp_pred_label.append(task_pred_label)
                    temp_pred_name.append(task_pred_name)
                    temp_pred_occ.append(task_pred_occ)

                pred_label_test.append(list(np.squeeze(temp_pred_label)))
                pred_names_test.append(list(np.squeeze(temp_pred_name)))
                pred_occ_test.append(np.squeeze(temp_pred_occ))
                pred_top_k.append(np.asarray(temp_pred_list))

        return pred_label_test, pred_names_test, pred_occ_test, pred_top_k

    def kNN_classification(self, task_predictions, task_name, label2class_list):
        if self.multiLabelsListOfVariables == None:
            task_pred_label, task_pred_name, task_pred_occ = self.single_label_kNN_classification(task_predictions)
        else:
            if task_name not in self.multiLabelsListOfVariables:
                task_pred_label, task_pred_name, task_pred_occ = self.single_label_kNN_classification(task_predictions)
            else:
                task_pred_label, task_pred_name, task_pred_occ = self.multi_label_kNN_classification(task_predictions,
                                                                                                     label2class_list)
        return task_pred_label, task_pred_name, task_pred_occ

    def multi_label_kNN_classification(self, task_predictions, label2class_list):
        if list(task_predictions).count('nan') == self.num_neighbors:
            task_pred_label = 'nan'
            task_pred_name = task_pred_label
            task_pred_occ = self.num_neighbors
        else:
            # get on-off label for every class for all kNN
            binary_pred_ind = []
            list_class_names = np.asarray(label2class_list[1:])
            for pred_label in task_predictions:
                pr_binary = [1 if temp_class in pred_label.split("___") else 0 for temp_class in list_class_names]
                binary_pred_ind.append(pr_binary)

            # binary decision, whether class shall be selected
            class_prediction = ""
            for label_ind, pot_label in enumerate(list_class_names):
                if stats.mode(np.asarray(binary_pred_ind)[:, label_ind])[0][0] == 1:
                    class_prediction = class_prediction + pot_label + "___"

            # combine all selected classes to a prediction
            # handle nan-prediction case with "nan_OR_" + most probable class
            if class_prediction == "":
                task_pred_label = "nan_OR_" + stats.mode(task_predictions)[0][0]
            else:
                task_pred_label = class_prediction[0:-3]

            cleaned_pred_list = list(task_predictions[task_predictions != 'nan'])
            task_pred_occ = stats.mode(cleaned_pred_list)[1][0]

        return task_pred_label, task_pred_label, task_pred_occ

    def single_label_kNN_classification(self, task_predictions):
        if list(task_predictions).count('nan') == self.num_neighbors:
            task_pred_label = 'nan'
            task_pred_name = task_pred_label
            task_pred_occ = self.num_neighbors
        else:
            cleaned_pred_list = list(task_predictions[task_predictions != 'nan'])
            task_pred_label = stats.mode(cleaned_pred_list)[0][0]
            task_pred_name = task_pred_label
            task_pred_occ = stats.mode(cleaned_pred_list)[1][0]
            # TODO: handle this case:
            # if task_pred_occ == int(len(cleaned_pred_list)/2):
            #     print(cleaned_pred_list)
            #     print(stats.mode(cleaned_pred_list))
            #     print(task_pred_name)
        return task_pred_label, task_pred_name, task_pred_occ

    def _write_knn_list_and_lut(self, query_image_list, tree_image_list, labels_target, pred_names_test, labels_tree,
                                tree_indices_knn, tree_dist_knn, relevant_variables):
        # Save GT, Prediction, names of k nearest neighbors and their distances to a text file
        all_used_images_test = [os.path.relpath(file) for file in query_image_list]
        knn_image_names = []
        knn_file = open(os.path.abspath(self.pred_gt_dir + '/' + 'knn_list.txt'), 'w')
        # TODO: D4.6: Güte der Klass. bei Pred. mit angeben (Anzahl occ)
        for idx in (range(np.shape(tree_indices_knn)[0])):
            # write query image
            temp_knn_image_names = []
            knn_file.write("*******%s*******" % (all_used_images_test[idx]))

            # write ground truth
            if self.bool_labeled_input:
                knn_file.write("\n Groundtruth: \n")
                for cur_label in relevant_variables:
                    knn_file.write("#%s \t" % cur_label)
                knn_file.write("\n")
                for gt in range(np.shape(pred_names_test)[1]):
                    knn_file.write("%s \t \t" % (np.asarray(labels_target)[idx, gt]))

            # write predictions
            knn_file.write("\n Predictions: \n")
            for cur_label in relevant_variables:
                knn_file.write("#%s \t" % cur_label)
            knn_file.write("\n")
            for gt in range(np.shape(pred_names_test)[1]):
                knn_file.write("%s \t \t" % (np.asarray(pred_names_test)[idx, gt]))

            # write knn
            knn_file.write("\n k nearest neighbours: \n")
            knn_file.write("#filename \t #distance ")
            for cur_label in relevant_variables:
                knn_file.write("#%s \t" % cur_label)
            knn_file.write("\n")
            for knn in range(np.shape(tree_indices_knn)[1]):
                # names of nn
                knn_file.write("%s \t" % (np.asarray(tree_image_list)[tree_indices_knn[idx][knn]]))

                # descriptor distance of knn
                knn_file.write("%s \t" % (np.asarray(tree_dist_knn)[idx][knn]))
                temp_knn_image_names.append(np.asarray(tree_image_list)[tree_indices_knn[idx][knn]])

                # labels of nn
                if len(relevant_variables) > 1:
                    for gt in range(np.shape(pred_names_test)[1]):
                        knn_file.write("%s \t" % (np.asarray(labels_tree)[tree_indices_knn[idx][knn], gt]))
                elif len(relevant_variables) == 1:
                    knn_file.write("%s \t" % np.asarray(labels_tree)[tree_indices_knn[idx][knn]])
                knn_file.write("\n")
            knn_file.write("\n")

            knn_image_names.append(temp_knn_image_names)
        knn_file.close()

        # select and store data for CSV LUT
        lut_knn_dict = {"input_image_name": query_image_list,
                        "kNN_image_names": knn_image_names,
                        "kNN_kg_object_uri": [[i.split("\\")[-1].split("__")[1] for i in j[:]] for j in
                                              knn_image_names],
                        "kNN_kD_index": [str(list(i)) for i in tree_indices_knn],
                        "kNN_descriptor_dist": [str(list(i)) for i in tree_dist_knn]}
        lut_df = pd.DataFrame(lut_knn_dict)
        lut_df.to_csv(os.path.join(self.pred_gt_dir, "kNN_LUT.csv"), index=False)
        # copy_kNN_images_to_one_folder(LUT_kNN_dict["input_image_name"],
        #                               LUT_kNN_dict["kNN_image_names"],
        #                               LUT_kNN_dict["input_kg_object_uri"],
        #                               dist,
        #                               savepath)

    """ -------------------------------- Routines data --------------------------------------------------------------"""

    def pre_compute_colour_descriptors(self):
        # colour distributions are computed for the "original images", i.e. smaller side = 448 pixel
        # -> gets the path to the "original images"

        # check if pre-computed colour distributions exist
        if os.path.isfile("./samples/pre_computed_hs_counter_resolution_" +
                          str(self.colour_distr_resolution) + ".npz"):
            pre_computed_hs_counter = np.load("./samples/pre_computed_hs_counter_resolution_" +
                                              str(self.colour_distr_resolution) + ".npz",
                                              allow_pickle=True)["arr_0"].item()
        else:
            pre_computed_hs_counter = {}

        # load or compute colour distribution into the according variable
        self.all_image_hs_counter = {}
        print('\n\nExtracting colour information (all images)...\t')
        for img in tqdm(self.all_full_image_names):
            if os.path.normpath(img) in pre_computed_hs_counter.keys():
                cur_hs_counter = pre_computed_hs_counter[img]
            else:
                name = os.path.basename(img)
                path = os.path.dirname(img)
                cur_hs_counter = hue_sat.get_polar_counter_from_img_file(path,
                                                                         name,
                                                                         self.colour_distr_resolution)
                pre_computed_hs_counter[os.path.normpath(img)] = cur_hs_counter
            self.all_image_hs_counter[img] = cur_hs_counter

        np.savez("./samples/pre_computed_hs_counter_resolution_" +
                 str(self.colour_distr_resolution) + ".npz", pre_computed_hs_counter)

    def structure_and_load_image_data(self):
        if self.bool_domain_classifier:
            relevant_var = self.relevant_variables[:]
            relevant_var.append(self.domain_variable)
        else:
            relevant_var = self.relevant_variables

        # semantic sample_handler for:
        #   - semantic similarity computation
        #   - augment and colour?
        self.sample_handler = SampleHandler(masterfile_dir=self.master_file_dir,
                                            masterfile_name=self.master_file_name,
                                            relevant_variables=relevant_var,
                                            image_based_samples=self.image_based_samples,
                                            validation_percentage=self.validation_percentage,
                                            multiLabelsListOfVariables=self.multiLabelsListOfVariables)
        self.all_full_image_names = np.concatenate((list(self.sample_handler.data_all['train'].keys()),
                                                    list(self.sample_handler.data_all['valid'].keys())),
                                                   axis=0)
        if self.loss_ind == "combined_similarity_loss":
            # rule sample_handler for:
            #   - rule similarity computation
            #   - augment and colour?
            if self.rules_batch_size_fraction != 0:
                self.sample_handler_rules = SampleHandler(masterfile_dir=self.master_file_dir,
                                                          masterfile_name=self.master_file_rules_name,
                                                          relevant_variables=relevant_var,
                                                          image_based_samples=self.image_based_samples,
                                                          validation_percentage=self.validation_percentage,
                                                          multiLabelsListOfVariables=self.multiLabelsListOfVariables,
                                                          bool_unlabeled_dataset_rules=True,
                                                          max_num_classes_per_variable_default=
                                                          self.sample_handler.max_num_classes_per_variable)
                all_img_names_rules = np.concatenate((list(self.sample_handler_rules.data_all['train'].keys()),
                                                      list(self.sample_handler_rules.data_all['valid'].keys())),
                                                     axis=0)
                self.all_full_image_names = np.concatenate((self.all_full_image_names, all_img_names_rules), axis=0)

            # additional sample_handler for
            #   - augment and colour similarity
            if self.colour_augment_batch_size_fraction != 0:
                self.sample_handler_colour_augment = SampleHandler(masterfile_dir=self.master_file_dir,
                                                                   masterfile_name=self.master_file_colour_augment_name,
                                                                   relevant_variables=relevant_var,
                                                                   image_based_samples=self.image_based_samples,
                                                                   validation_percentage=self.validation_percentage,
                                                                   multiLabelsListOfVariables=
                                                                   self.multiLabelsListOfVariables)
                all_img_names_colour_augment = np.concatenate(
                    (list(self.sample_handler_colour_augment.data_all['train'].keys()),
                     list(self.sample_handler_colour_augment.data_all['valid'].keys())),
                    axis=0)
                self.all_full_image_names = np.concatenate((self.all_full_image_names, all_img_names_colour_augment),
                                                           axis=0)
            
            if self.loss_weight_rules != 0:
                # create indicator LUT (similar, dissimilar) from rules
                self.load_all_indicators_from_rules()

    def get_colour_correlation_batch(self, train_image_name):
        if self.loss_ind in ["hue_saturation_similarity_loss", "combined_similarity_loss"] \
                and self.loss_weight_colour > 0:
            hs_counter_batch = []
            for im_file in train_image_name:
                hs_counter_batch.append(self.all_image_hs_counter[im_file])
            batch_norm_cross_corr = hue_sat.get_normalized_cross_correlation_batch(hs_counter_batch)
        else:
            batch_norm_cross_corr = 1e-16 + np.zeros((len(train_image_name), len(train_image_name)))
        return batch_norm_cross_corr

    def load_all_indicators_from_rules(self):
        # 1. get all images from the master file of the labeled datatset and of the rules dataset
        #    i.e. all image names that will be used in the subsequent code
        #    -> already contained in self.sample_handler, self.sample_handler_rules
        self.all_base_image_names = [os.path.basename(image) for image in self.all_full_image_names]

        # 2. create data structures for similar and dissimilar indicators filled with zeros
        #    one entry per possible image pair
        self.all_indicator_similar_rules = np.zeros((len(self.all_base_image_names),
                                                     len(self.all_base_image_names)))
        self.all_indicator_dissimilar_rules = np.zeros(
            (len(self.all_base_image_names), len(self.all_base_image_names)))

        # 3. load similar and fill data structure
        if os.path.isfile(os.path.join(self.master_file_dir, self.master_file_similar_obj)):
            # load objects of similar rules
            similar_rules_dict = self.get_rules_dict(rules_master_file=self.master_file_similar_obj)
            # fill data structure according to similar rules files
            for _, obj_list in similar_rules_dict.items():
                self.all_indicator_similar_rules = self.update_indicator_matrix_according_to_rule(
                    current_rules_objects=obj_list,
                    matrix_2_be_updated=self.all_indicator_similar_rules)
            # fill data structure according to multi-image objects
            all_obj_uris = [img.split("__")[1] for img in self.all_base_image_names]
            all_obj_uris, obj_uri_count = np.unique(all_obj_uris, return_counts=True)
            for (uri, count) in zip(all_obj_uris, obj_uri_count):
                if count > 1:
                    self.all_indicator_similar_rules = self.update_indicator_matrix_according_to_rule(
                        current_rules_objects=[uri],
                        matrix_2_be_updated=self.all_indicator_similar_rules)

        # 4. load dissimilar and fill data structure
        if os.path.isfile(os.path.join(self.master_file_dir, self.master_file_dissimilar_obj)):
            # load objects of dissimilar rules
            dissimilar_rules_dict = self.get_rules_dict(rules_master_file=self.master_file_dissimilar_obj)

            # fill data structure
            for _, obj_list in dissimilar_rules_dict.items():
                self.all_indicator_dissimilar_rules = self.update_indicator_matrix_according_to_rule(
                    current_rules_objects=obj_list,
                    matrix_2_be_updated=self.all_indicator_dissimilar_rules)
        else:
            print("Rule-based loss shall be minimized, but no examples for dissimilar examples were provided!")

    def update_indicator_matrix_according_to_rule(self, current_rules_objects, matrix_2_be_updated):
        all_rule_related_img = [img for img in self.all_base_image_names if
                                img.split("__")[1] in current_rules_objects]
        if len(all_rule_related_img) > 0:
            ind_rule_img = [ind for ind, img in enumerate(self.all_base_image_names) if
                            img in all_rule_related_img]
            for ind_one in ind_rule_img:
                for ind_two in ind_rule_img:
                    if ind_one != ind_two:
                        matrix_2_be_updated[ind_one, ind_two] = 1.
        return matrix_2_be_updated

    def get_rules_dict(self, rules_master_file):
        rules_files_list = wp4lib.master_file_to_collections_list(master_dir=self.master_file_dir,
                                                                  master_file_name=rules_master_file)
        rules_dict = {}
        for rules_file in rules_files_list:
            rules_objects = pd.read_csv(os.path.join(self.master_file_dir, rules_file), header=None)
            rules_dict[rules_file] = [obj_uri.split("/")[-1] for obj_uri in list(rules_objects[0])]
        return rules_dict

    def get_rules_indicator_batch(self, image_name_batch):
        # print("train_image_name", len(image_name_batch))
        if self.loss_weight_rules > 0:
            image_name_batch = [os.path.basename(image) for image in image_name_batch]
            ind_batch_img = [ind for ind, img in enumerate(self.all_base_image_names) if
                             img in image_name_batch]
            # print("ind_batch_img", len(ind_batch_img))
            # c, v = np.unique(image_name_batch, return_counts=True)
            # print("unique", max(v))
            rules_indicator_similar_values = self.get_indicator_submatrix(
                master_matrix=self.all_indicator_similar_rules,
                ind_for_submat=ind_batch_img)
            rules_indicator_dissimilar_values = self.get_indicator_submatrix(
                master_matrix=self.all_indicator_dissimilar_rules,
                ind_for_submat=ind_batch_img)
        else:
            rules_indicator_similar_values = 1e-16 + np.zeros((len(image_name_batch), len(image_name_batch)))
            rules_indicator_dissimilar_values = 1e-16 + np.zeros((len(image_name_batch), len(image_name_batch)))
        return rules_indicator_similar_values, rules_indicator_dissimilar_values

    def get_indicator_submatrix(self, master_matrix, ind_for_submat):
        submatrix = master_matrix[np.ix_(ind_for_submat, ind_for_submat)]
        return submatrix

    def create_batch(self, augmented_image_tensor, in_img_tensor, jpeg_data_tensor, sess, data_creation_purpose):
        batch_size = 0
        if data_creation_purpose == 'train':
            batch_size = self.batch_size
            batch_size_semantic = int(batch_size * self.semantic_batch_size_fraction)
            batch_size_rules = int(batch_size * self.rules_batch_size_fraction)
            batch_size_colour_augment = int(batch_size * self.colour_augment_batch_size_fraction)
        elif data_creation_purpose == 'valid':
            # print("batch size valid", self.batch_size)
            # TODO: take whole batch and not number of samples in train batch
            #       -> needs iterative mean computation of validation loss
            batch_size = min(self.batch_size,
                             self.sample_handler.amountOfValidationSamples -
                             self.sample_handler.nextUnusedValidationSampleIndex)
            batch_size_semantic = min(batch_size,
                                      int(batch_size * self.semantic_batch_size_fraction))
            if self.rules_batch_size_fraction != 0:
                batch_size_rules = min(self.batch_size,
                                       int(batch_size * self.rules_batch_size_fraction),
                                       self.sample_handler_rules.amountOfValidationSamples -
                                       self.sample_handler_rules.nextUnusedValidationSampleIndex)
            if self.colour_augment_batch_size_fraction != 0:
                batch_size_colour_augment = min(self.batch_size,
                                                int(batch_size * self.colour_augment_batch_size_fraction),
                                                self.sample_handler_colour_augment.amountOfValidationSamples -
                                                self.sample_handler_colour_augment.nextUnusedValidationSampleIndex)
        else:
            assert data_creation_purpose in ["train", "valid"], \
                'No valid data_creation_purpose.'

        # semantic similarity loss only
        if self.loss_ind != "combined_similarity_loss":
            (image_data,
             ground_truth,
             image_name) = self.sample_handler.get_random_samples(
                how_many=batch_size,
                purpose=data_creation_purpose,
                session=sess,
                jpeg_data_tensor=jpeg_data_tensor,
                decoded_image_tensor=in_img_tensor)
            if self.bool_domain_classifier:
                ground_truth_domain = list(np.asarray(ground_truth)[-1, :])
                ground_truth = list(np.asarray(ground_truth)[0:-1, :])
            else:
                ground_truth_domain = []
        # combined similarity loss only
        # thus, implicitly all other losses via loss_term_weights
        else:
            image_data = []
            ground_truth = []
            image_name = []
            # create a batch that theoretically can be used for all loss terms
            # practically: may no images with a rule are included
            if self.semantic_batch_size_fraction != 0:
                (image_data,
                 ground_truth,
                 image_name) = self.sample_handler.get_random_samples(
                    how_many=batch_size_semantic,
                    purpose=data_creation_purpose,
                    session=sess,
                    jpeg_data_tensor=jpeg_data_tensor,
                    decoded_image_tensor=in_img_tensor)
            # print("batch_size_semantic", batch_size_semantic)
            # print("image_name", len(np.unique(image_name)))
            # in case that the dataset shall be expanded by unlabeled samples having a rule
            # (self.rules_batch_size_fraction != 0)
            # which only make sense, if the rules loss shall be calculated
            # (self.loss_weight_rules != 0)
            if self.rules_batch_size_fraction != 0:
                (image_data_rules,
                 ground_truth_rules,
                 image_name_rules) = self.sample_handler_rules.get_random_samples(
                    how_many=batch_size_rules,
                    purpose=data_creation_purpose,
                    session=sess,
                    jpeg_data_tensor=jpeg_data_tensor,
                    decoded_image_tensor=in_img_tensor)
                # print("batch_size_rules", batch_size_rules)
                # print("image_name_rules", len(np.unique(image_name_rules)))
                if image_name:
                    image_data = np.concatenate((image_data, image_data_rules),
                                                axis=0)
                    ground_truth = np.concatenate((ground_truth, ground_truth_rules),
                                                  axis=0)
                    image_name = np.concatenate((image_name, image_name_rules),
                                                axis=0)
                else:
                    image_data = image_data_rules
                    ground_truth = ground_truth_rules
                    image_name = image_name_rules

            # in case that the dataset shall be expanded by unlabeled samples wihtout a rule
            # (self.colour_augment_batch_size_fraction != 0)
            # which only make sense, if the either the colour loss...
            # (self.loss_weight_colour != 0)
            # ... or the self augmentation loss will be claculated.
            # (self.loss_weight_augment != 0)
            if (self.loss_weight_colour != 0 or self.loss_weight_augment != 0) and \
                    self.colour_augment_batch_size_fraction != 0:
                (image_data_colour_augment,
                 ground_truth_colour_augment,
                 image_name_colour_augment) = self.sample_handler_colour_augment.get_random_samples(
                    how_many=batch_size_colour_augment,
                    purpose=data_creation_purpose,
                    session=sess,
                    jpeg_data_tensor=jpeg_data_tensor,
                    decoded_image_tensor=in_img_tensor)
                if image_name:
                    image_data = np.concatenate((image_data, image_data_colour_augment),
                                                axis=0)
                    ground_truth = np.concatenate((ground_truth, ground_truth_colour_augment),
                                                  axis=0)
                    image_name = np.concatenate((image_name, image_name_colour_augment),
                                                axis=0)
                else:
                    image_data = image_data_colour_augment
                    ground_truth = ground_truth_colour_augment
                    image_name = image_name_colour_augment

            # in any (single-label) case, a domain loss can be added
            if self.bool_domain_classifier:
                # TODO: Handle domain_ground_truth in domain loss for multi-label case (1-hot)
                ground_truth_domain = list(np.asarray(ground_truth)[-1, :])
                ground_truth = list(np.asarray(ground_truth)[0:-1, :])
            else:
                ground_truth_domain = []

        if data_creation_purpose == "train":
            # Online Data Augmentation for all images in the resulting batch
            # TODO: incorporate "elastic distortions" (https://github.com/mdbloice/Augmentor)
            vardata = [sess.run(augmented_image_tensor,
                                feed_dict={in_img_tensor: img_data}) for img_data in image_data]
            batch_in_img = vardata
        else:
            batch_in_img = image_data

        return batch_in_img, ground_truth, ground_truth_domain, image_name

    @staticmethod
    def create_config_file_train_model(master_file_path, variable_list):
        config = open("Configuration_train_model.txt", "w+")

        config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
        config.writelines(["master_file_name; master_file_tree.txt\n"])
        config.writelines(["master_file_dir; " + master_file_path + "\n"])
        config.writelines(["log_dir; " + r"./output_files/Default/log_dir/" + "\n"])
        config.writelines(["model_dir; " + r"./output_files/Default/model_dir/" + "\n"])

        config.writelines(["\n****************CNN ARCHITECTURE SPECIFICATIONS**************** \n"])
        config.writelines(["add_fc; [1024, 128] \n"])
        config.writelines(["num_fine_tune_layers; 0 \n"])

        config.writelines(["\n****************TRAINING SPECIFICATIONS**************** \n"])
        config.writelines(["batch_size; 150\n"])
        config.writelines(["how_many_training_steps; 200\n"])
        config.writelines(["learning_rate; 1e-4\n"])
        config.writelines(["validation_percentage; 25\n"])
        config.writelines(["how_often_validation; 10\n"])
        config.writelines(["loss_ind; soft_triplet_loss\n"])

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

    @staticmethod
    def create_config_file_build_tree(master_file_path, variable_list):
        config = open("Configuration_build_kDTree.txt", "w+")

        config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
        config.writelines(["model_dir; " + r"./output_files/Default/model_dir/" + "\n"])
        config.writelines(["master_file_tree; master_file_tree.txt\n"])
        config.writelines(["master_dir_tree; " + master_file_path + "\n"])
        config.writelines(["tree_dir; " + r"./output_files/Default/tree_dir/" + "\n"])

        config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
        config.writelines(["relevant_variables; "])
        for variable in variable_list[0:-2]:
            config.writelines(["#%s, " % str(variable)])
        config.writelines(["#%s" % str(variable_list[-2])])
        config.close()

    @staticmethod
    def create_config_file_get_knn(master_file_path):
        config = open("Configuration_get_kNN.txt", "w+")

        config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
        config.writelines(["tree_dir; " + r"./output_files/Default/tree_dir/" + "\n"])
        config.writelines(["master_file_retrieval; master_file_prediction.txt\n"])
        config.writelines(["master_dir_retrieval; " + master_file_path + "\n"])
        config.writelines(["model_dir; " + r"./output_files/Default/model_dir/" + "\n"])
        config.writelines(["pred_gt_dir; " + r"./output_files/Default/pred_gt_dir/" + "\n"])

        config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
        config.writelines(["num_neighbors; 6\n"])
        config.writelines(["bool_labeled_input; True\n"])
        config.close()

    @staticmethod
    def create_config_file_evaluate_model():
        """

        """
        config = open("Configuration_evaluate_model.txt", "w+")

        config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
        config.writelines(["pred_gt_dir; " + r"./output_files/Default/pred_gt_dir/" + "\n"])
        config.writelines(["eval_result_dir; " + r"./output_files/Default/eval_result_dir/" + "\n"])
        config.close()

    @staticmethod
    def create_config_file_cross_validation(master_file_path, variable_list):
        """

        """
        config = open("Configuration_crossvalidation.txt", "w+")

        config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
        config.writelines(["master_file_name; masterfile.txt\n"])
        config.writelines(["master_file_dir; " + master_file_path + "\n"])
        config.writelines(["log_dir; " + r"./output_files/Default/log_dir/" + "\n"])
        config.writelines(["model_dir; " + r"./output_files/Default/model_dir/" + "\n"])
        config.writelines(["tree_dir; " + r"./output_files/Default/tree_dir/" + "\n"])
        config.writelines(["pred_gt_dir; " + r"./output_files/Default/pred_gt_dir/" + "\n"])
        config.writelines(["eval_result_dir; " + r"./output_files/Default/eval_result_dir/" + "\n"])

        config.writelines(["\n****************CNN ARCHITECTURE SPECIFICATIONS**************** \n"])
        config.writelines(["add_fc; [1024, 128] \n"])
        config.writelines(["num_fine_tune_layers; 0 \n"])

        config.writelines(["\n****************TRAINING SPECIFICATIONS**************** \n"])
        config.writelines(["batch_size; 150\n"])
        config.writelines(["how_many_training_steps; 100\n"])
        config.writelines(["learning_rate; 1e-4\n"])
        config.writelines(["validation_percentage; 25\n"])
        config.writelines(["how_often_validation; 10\n"])
        config.writelines(["loss_ind; soft_triplet_loss\n"])

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

    def _write_train_parameters_to_configuration_file(self):
        config = open(os.path.join(self.model_dir, "Configuration_train_model.txt"), "w")

        config.writelines(["****************FILES AND DIRECTORIES**************** \n"])
        config.writelines(["master_file_name; " + self.master_file_name + "\n"])
        config.writelines(["master_file_dir; " + self.master_file_dir + "\n"])
        config.writelines(["log_dir; " + self.log_dir + "\n"])
        config.writelines(["model_dir; " + self.model_dir + "\n"])

        config.writelines(["\n****************CNN ARCHITECTURE SPECIFICATIONS**************** \n"])
        config.writelines(["add_fc; " + str(self.add_fc) + "\n"])
        config.writelines(["num_fine_tune_layers; " + str(self.num_fine_tune_layers) + "\n"])

        config.writelines(["\n****************TRAINING SPECIFICATIONS**************** \n"])
        config.writelines(["batch_size; " + str(self.batch_size) + "\n"])
        config.writelines(["how_many_training_steps; " + str(self.how_many_training_steps) + "\n"])
        config.writelines(["learning_rate; " + str(self.learning_rate) + "\n"])
        config.writelines(["validation_percentage; " + str(self.validation_percentage) + "\n"])
        config.writelines(["how_often_validation; " + str(self.how_often_validation) + "\n"])
        config.writelines(["loss_ind; " + self.loss_ind + "\n"])

        config.writelines(["\n****************SIMILARITY SPECIFICATIONS**************** \n"])
        config.writelines(["relevant_variables; "])
        for variable in self.relevant_variables[0:-1]:
            config.writelines(["#%s, " % str(variable)])
        config.writelines(["#%s\n" % str(self.relevant_variables[-1])])

        config.writelines(["\n****************DATA AUGMENTATION SPECIFICATIONS**************** \n"])
        config.writelines(["random_crop; " + str(self.aug_set_dict["random_crop"]) + "\n"])
        config.writelines(["random_rotation90; " + str(self.aug_set_dict["random_rotation90"]) + "\n"])
        config.writelines(["gaussian_noise; " + str(self.aug_set_dict["gaussian_noise"]) + "\n"])
        config.writelines(["flip_left_right; " + str(self.aug_set_dict["flip_left_right"]) + "\n"])
        config.writelines(["flip_up_down; " + str(self.aug_set_dict["flip_up_down"]) + "\n"])
        config.close()

    def _copy_collections(self):
        coll_list = wp4lib.master_file_to_collections_list(self.master_file_dir, self.master_file_name)
        for collection in coll_list:
            copy(os.path.join(self.master_file_dir, collection),
                 os.path.join(self.model_dir, collection))

    """ -------------------------------- Routines Config file -------------------------------------------------------"""

    def read_configfile(self, configfile):
        """Reads all types of configfiles and sets internal parameters"""

        control_id = open(configfile, 'r', encoding='utf-8')
        for variable in control_id:

            """------------------ Create Dataset --------------------------------------------------------------------"""
            if variable.split(';')[0] == 'csvfile':
                self.csv_file = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'imgsave_dir':
                self.img_save_dir = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'minnumsamples':
                self.min_samples_class = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'retaincollections':
                self.retain_collections = variable.split(';')[1] \
                    .replace(' ', '').replace('\n', '') \
                    .replace('\t', '').split(',')

            if variable.split(';')[0] == 'num_labeled':
                self.num_labeled = int(variable.split(';')[1].strip())

            """------------------ Directories -----------------------------------------------------------------------"""
            if variable.split(';')[0] == 'master_file_name':
                self.master_file_name = variable.split(';')[1].strip()
                self.master_file_name_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'master_file_dir':
                self.master_file_dir = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'master_file_tree':
                self.master_file_tree = variable.split(';')[1].strip()
                self.master_file_tree_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'master_dir_tree':
                self.master_dir_tree = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'master_file_retrieval':
                self.master_file_retrieval = variable.split(';')[1].strip()
                self.master_file_retrieval_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'master_dir_retrieval':
                self.master_dir_retrieval = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'log_dir':
                self.log_dir = variable.split(';')[1].strip()
                self.log_dir_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'eval_result_dir':
                self.eval_result_dir = variable.split(';')[1].strip()
                self.eval_result_dir_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'model_dir':
                self.model_dir = variable.split(';')[1].strip()
                self.model_dir_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'pred_gt_dir':
                self.pred_gt_dir = variable.split(';')[1].strip()
                self.pred_gt_dir_cv = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'tree_dir':
                self.tree_dir = variable.split(';')[1].strip()
                self.tree_dir_cv = variable.split(';')[1].strip()

            """------------------ Network Architecture --------------------------------------------------------------"""
            if variable.split(';')[0] == 'num_joint_fc_layer':
                self.num_joint_fc_layer = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'num_nodes_joint_fc':
                self.num_nodes_joint_fc = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'num_fine_tune_layers':
                self.num_fine_tune_layers = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'relevant_variables':
                self.relevant_variables = variable.split(';')[1].replace(',', '') \
                                              .replace(' ', '').replace('\n', '') \
                                              .replace('\t', '').split('#')[1:]
                self.variable_weights = list(np.ones(len(self.relevant_variables)))

            if variable.split(';')[0] == 'add_fc':
                if len(variable.split('[')[1].split(']')[0]) > 0:
                    self.add_fc = list(map(int, variable.split('[')[1].split(']')[0].split(',')))
                else:
                    self.add_fc = []

            """------------------ Training Specifications -----------------------------------------------------------"""
            if variable.split(';')[0] == 'batch_size':
                self.batch_size = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'how_many_training_steps':
                self.how_many_training_steps = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'how_often_validation':
                self.how_often_validation = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'validation_percentage':
                self.validation_percentage = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'learning_rate':
                self.learning_rate = np.float(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'weight_decay':
                self.weight_decay = float(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'min_samples_per_class':
                self.min_samples_per_class = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'num_task_stop_gradient':
                self.num_task_stop_gradient = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'min_num_labels':
                self.min_num_labels = int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'optimizer_ind':
                self.optimizer_ind = variable.split(';')[1].strip()

            if variable.split(';')[0] == 'loss_ind':
                self.loss_ind = variable.split(';')[1].strip()

            """------------------ Retrieval Specifications ----------------------------------------------------------"""
            if variable.split(';')[0] == 'num_neighbors':
                self.num_neighbors = np.int(variable.split(';')[1].strip())

            if variable.split(';')[0] == 'bool_labeled_input':
                self.bool_labeled_input = variable.split(';')[1].strip()
                if self.bool_labeled_input == 'True':
                    self.bool_labeled_input = True
                else:
                    self.bool_labeled_input = False

            """------------------ Augmentation ----------------------------------------------------------------------"""
            if variable.split(';')[0] == 'flip_left_right':
                flip_left_right = variable.split(';')[1].strip()
                if flip_left_right == 'True':
                    flip_left_right = True
                else:
                    flip_left_right = False
                self.aug_set_dict['flip_left_right'] = flip_left_right

            if variable.split(';')[0] == 'flip_up_down':
                flip_up_down = variable.split(';')[1].strip()
                if flip_up_down == 'True':
                    flip_up_down = True
                else:
                    flip_up_down = False
                self.aug_set_dict['flip_up_down'] = flip_up_down

            if variable.split(';')[0] == 'random_shear':
                random_shear = list(map(float,
                                        variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_shear'] = random_shear

            if variable.split(';')[0] == 'random_brightness':
                random_brightness = int(variable.split(';')[1].strip())
                self.aug_set_dict['random_brightness'] = random_brightness

            if variable.split(';')[0] == 'random_crop':
                random_crop = list(map(float,
                                       variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_crop'] = random_crop

            if variable.split(';')[0] == 'random_rotation':
                random_rotation = float(variable.split(';')[1].strip()) * math.pi / 180
                self.aug_set_dict['random_rotation'] = random_rotation

            if variable.split(';')[0] == 'random_contrast':
                random_contrast = list(map(float,
                                           variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_contrast'] = random_contrast

            if variable.split(';')[0] == 'random_hue':
                random_hue = float(variable.split(';')[1].strip())
                self.aug_set_dict['random_hue'] = random_hue

            if variable.split(';')[0] == 'random_saturation':
                random_saturation = list(map(float,
                                             variable.split('[')[1].split(']')[0].split(',')))
                self.aug_set_dict['random_saturation'] = random_saturation

            if variable.split(';')[0] == 'random_rotation90':
                random_rotation90 = variable.split(';')[1].strip()
                if random_rotation90 == 'True':
                    random_rotation90 = True
                else:
                    random_rotation90 = False
                self.aug_set_dict['random_rotation90'] = random_rotation90

            if variable.split(';')[0] == 'gaussian_noise':
                gaussian_noise = float(variable.split(';')[1].strip())
                self.aug_set_dict['gaussian_noise'] = gaussian_noise

        control_id.close()
