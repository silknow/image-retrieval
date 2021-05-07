# -*- coding: utf-8 -*-
"""
Created on Wed July 15 11:43:10 2020

@author: dorozynski
"""
import sys
import tensorflow as tf
import numpy as np
import math


class SimilarityLosses():
    # TODO: consider label_weight_tensor
    """

    """

    def __init__(self, relevant_variables, in_label_tensor, output_feature_tensor,
                 label_weight_tensor, similarity_thresh, loss_ind, batch_size,
                 output_feature_tensor_augmented, rules_indicator_similar, rules_indicator_dissimilar,
                 norm_cross_corr_plh, weights_combined_loss,
                 margin=2, colour_distr_resolution=5, multiLabelsListOfVariables=None):
        """Initializes a similarity loss class.

        :Arguments\::
            :relevant_variables (*list of strings*)\::
                All variables that are considered for semantic similarity.
            :in_label_tensor (*tensor*)\::
                Tensor that will be fed in training with the reference labels for the variables.
                size = batch_size x len(relevant_variables). Contains indices of the class labels for the
                variables. If unknown, label = -1".
            :output_feature_tensor (*tensor*)\::
                size = batch_size x num_features. Tensor with feature vectors for all samples
                in the batch
            :label_weight_tensor (*tensor*)\::
                size = len(relevant_variables). Contains importance weights for the variables
                according to their importance in the semantic similarity calculation.
                NOT USED NOW.
            :similarity_thresh (*float*)\::
                Threshold for similarity in hard loss variants (old implementations in Git).
                NOT USED NOW.
            :loss_ind (*string*)\::
                Indicator for the type of similarity loss to be used.
            :output_feature_tensor_augmented (*tensor*)\::
                Only for self augmentation loss. contains feature vectors of self augmentations.
            :rules_indicator_similar (*tensor*)\::
                size = batch-size x batch-size. Contains indicators in entry (i, j) whether the
                i-th and j-th sample in a batch are similar (value = 1) or whether no knowledge
                is available (value = 0) according to cultural heritage domain
                experts.
            :rules_indicator_dissimilar (*tensor*)\::
                size = batch-size x batch-size. Contains indicators in entry (i, j) whether the
                i-th and j-th sample in a batch are dissimilar (value = 1) or whether no knowledge
                is available (value = 0) according to cultural heritage domain.
            :norm_cross_corr_plh (*tensor of floats*)\::
                size = batch-size x batch-size. Contains the normalized cross correlation
                coefficients of the hue-saturation-distributions. For colour similarity loss only.
            :weights_combined_loss (*tensor*)\::
                size = (4,). Contains the weights for the four loss terms where the order is:
                    1. semantic weight
                    2. rules weight
                    3. colour weight
                    4. self-augmentation weight.
            :multiLabelsListOfVariables (*list*)\::
                List of variable names (strings) for which multiple labels per variable are available.
                Default = None. If None, only single labels are available for all variables.

        """
        self.relevant_variables = relevant_variables  # semantic variables for semantic similarity
        self.in_label_tensor = in_label_tensor  # class labels for all samples for all variables
        self.output_feature_tensor = output_feature_tensor  # feature vectors for all samples
        self.label_weight_tensor = label_weight_tensor  # importance weights for semantic variables
        self.similarity_thresh = similarity_thresh  # threshold for hard variants of contrastive and triplet
        self.margin = margin  # unused!?
        self.loss_ind = loss_ind  # indicator for loss type
        self.batch_size = batch_size  # num of samples in batch
        # self.batch_img_tensor = batch_img_tensor  # images/samples
        self.colour_distr_resolution = colour_distr_resolution  # raster size for hue-saturation-distribution
        self.output_feature_tensor_augmented = output_feature_tensor_augmented  # dict with parameter setting for augmentation
        self.rules_indicator_similar = rules_indicator_similar  # len(output_feature_tensor) x len(output_feature_tensor)
        # 1: similar, 0 not similar
        self.rules_indicator_dissimilar = rules_indicator_dissimilar  # len(output_feature_tensor) x len(output_feature_tensor)
        # 1: dissimilar, 0 not similar
        self.norm_cross_corr = norm_cross_corr_plh
        self.weights_combined_loss = weights_combined_loss
        self.multiLabelsListOfVariables = multiLabelsListOfVariables

    def setup_loss(self):
        # TODO: Unterscheidliche Anzahl von valid samples (je nach loss): Gewichtugn!?

        # TODO: Aufpassen, weil self_augment_loss nur max batch_size einträge als Vektor hat und als einziger Loss kene mat.
        if self.loss_ind == 'soft_triplet_incomp_loss_min_margin':
            # ====================================================
            (loss, _,
             bool_reduce,
             bool_only_bs_hardest) = self.soft_triplet_incomp_loss_min_margin()

        elif self.loss_ind == 'soft_contrastive_loss':
            # ===============================================
            print("Contrastive loss for complete or incomplete labels will be minimized.\n")
            # TODO: Gebe indices für batch_hard_list aus (wichtig für combined loss)
            (loss,
             bool_reduce,
             bool_only_bs_hardest) = self.soft_contrastive_loss_comp_and_incomp()

        elif self.loss_ind == 'soft_triplet_loss':
            # ===============================================
            print("Triplet loss for complete or incomplete labels will be minimized.\n")
            # TODO: Gebe indices für batch_hard_list aus (wichtig für combined loss)
            (loss, _,
             bool_reduce,
             bool_only_bs_hardest) = self.soft_triplet_loss_comp_and_incomp()

        elif self.loss_ind == 'hue_saturation_similarity_loss':
            print("Label independent colour loss will be minimized.\n")
            (loss,
             bool_reduce,
             bool_only_bs_hardest) = self.colour_similarity_loss_comp_and_incomp()

        elif self.loss_ind == 'self_augmented_similarity_loss':
            print("Label independent self augmentation loss will be minimized.\n")
            (loss,
             bool_reduce) = self.self_augmentation_loss()
            bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_augment")

        elif self.loss_ind == 'prior_knowledge_similarity_loss':
            print("Label independent prior knowledge loss will be minimized.\n")
            (loss,
             bool_reduce,
             bool_only_bs_hardest) = self.similarity_prior_knowledge_loss()

        elif self.loss_ind == 'combined_similarity_loss':
            print("Combined similarity loss will be minimized.\n")
            (loss,
             bool_reduce,
             bool_only_bs_hardest) = self.combined_similarity_loss()

        return loss, bool_reduce, bool_only_bs_hardest

    """ -------------------------------- Combined similarity loss ---------------------------------------------------"""

    def combined_similarity_loss(self):
        # semantic part (default: triplet)
        if self.weights_combined_loss[0] == 0:  # no semantic loss
            loss_semantic = 0
            bool_reduce_semantic = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_semantic")
            bool_only_bs_hardest_semantic = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_semantic")
        else:
            (loss_semantic, _,
             bool_reduce_semantic,
             bool_only_bs_hardest_semantic) = self.soft_triplet_loss_comp_and_incomp()

        # rules part
        if self.weights_combined_loss[1] == 0:  # no rules loss
            loss_rules = 0
            bool_reduce_rules = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_rules")
            bool_only_bs_hardest_rules = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_rules")
        else:
            (loss_rules,
             bool_reduce_rules,
             bool_only_bs_hardest_rules) = self.similarity_prior_knowledge_loss()

        # colour part
        if self.weights_combined_loss[2] == 0:  # no colour loss
            loss_colour = 0
            bool_reduce_colour = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_colour")
            bool_only_bs_hardest_colour = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_colour")
        else:
            (loss_colour,
             bool_reduce_colour,
             bool_only_bs_hardest_colour) = self.colour_similarity_loss_comp_and_incomp()

        # self-augment part
        if self.weights_combined_loss[3] == 0:  # no augment loss
            loss_augment = 0
            bool_reduce_augment = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_augment")
        else:
            # has exactly batch_size entries -> no bool_bs_hardest
            (loss_augment,
             bool_reduce_augment) = self.self_augmentation_loss()

        # TODO: Problem for validation when less than batch_size samples left
        # -> take len(loss_augment) as bacth_size?
        # TODO: What is with self-augmented loss for validation???
        # TODO: Handling of the same samples in every loss component?
        # TODO: Weights of the losses according codomain (Wertebereich)!
        combined_loss_vec = self.weights_combined_loss[0] * tf.reduce_mean(loss_semantic) + \
                            self.weights_combined_loss[1] * tf.reduce_mean(loss_rules) + \
                            self.weights_combined_loss[2] * tf.reduce_mean(loss_colour) + \
                            self.weights_combined_loss[3] * tf.reduce_mean(loss_augment)

        bool_reduce_mean_combined = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_combined")
        combined_loss = tf.cond(bool_reduce_mean_combined,
                                lambda: tf.reduce_mean(combined_loss_vec),
                                lambda: combined_loss_vec)

        bool_only_bs_hardest = [bool_only_bs_hardest_semantic,
                                bool_only_bs_hardest_rules,
                                bool_only_bs_hardest_colour]

        bool_reduce_mean = [bool_reduce_semantic,
                            bool_reduce_rules,
                            bool_reduce_colour,
                            bool_reduce_augment,
                            bool_reduce_mean_combined]

        tf.compat.v1.summary.scalar("combined_similarity_loss", tf.reduce_mean(combined_loss))

        return combined_loss, np.asarray(bool_reduce_mean), np.asarray(bool_only_bs_hardest)

    """ -------------------------------- Self augmentation loss -----------------------------------------------------"""

    def similarity_prior_knowledge_loss(self):

        distances = self._pairwise_distances()
        kronecker_sim = self.rules_indicator_similar
        kronecker_dissim = self.rules_indicator_dissimilar
        m_neg_prior = 2.
        m_pos_prior = 0.

        # rules loss
        prior_loss = kronecker_sim * tf.maximum(0., distances - m_pos_prior) + \
                     kronecker_dissim * tf.maximum(0., m_neg_prior - distances)

        # consider only hardest pairs
        # TODO: Ckeck mining! ...and mean computation
        bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_rules")
        loss_hardest = tf.cond(bool_only_bs_hardest,
                               lambda: tf.sort(
                                   tf.reshape(prior_loss,
                                              [tf.shape(prior_loss)[0] * tf.shape(prior_loss)[1]]),
                                   direction='DESCENDING')[0:self.batch_size],
                               lambda: tf.reshape(prior_loss,
                                                  [tf.shape(prior_loss)[0] * tf.shape(prior_loss)[1]]))

        # return loss vector or mean
        bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_rules")
        prior_loss = tf.cond(bool_reduce_mean,
                             lambda: tf.reduce_sum(loss_hardest) / (
                                         tf.reduce_sum(kronecker_sim) + tf.reduce_sum(kronecker_dissim)),
                             lambda: loss_hardest)

        tf.compat.v1.summary.scalar("rules_prior_knowledge_loss", tf.reduce_mean(prior_loss))

        return prior_loss, bool_reduce_mean, bool_only_bs_hardest

    """ -------------------------------- Self augmentation loss -----------------------------------------------------"""

    def self_augmentation_loss(self):
        # TODO: HIer können auch komplett ungelabelte Samples genutzt werden; ist mit aktuellem
        # samplehandling aber nicht möglich
        # TODO: Außerfdem dann auch berücksichtigung in den anderen Loss funktionen
        # TODO: Wenn augmentation loss in combined loss, dann müssen alle anderen batch hard haben, weil pairwise dist
        # nur batch_size groß sein kann bei self_augmentation_loss

        # pairwise_dist = self._pairwise_distances(squared=False) von sample und augmentation!!!
        pairwise_dist = self._pairwise_distances_from_inputs(self.output_feature_tensor,
                                                             self.output_feature_tensor_augmented,
                                                             False)

        # self augmentation loss
        # do not need a margin
        # -> would be the margin for positive similarity
        # -> always 0 as the sample and its augmentation have the same information
        augment_loss = tf.maximum(0., pairwise_dist)

        # # consider only hardest pairs
        # bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_augment")
        # loss_hardest = tf.cond(bool_only_bs_hardest,
        #                        lambda: tf.sort(
        #                            tf.reshape(augment_loss,
        #                                       [tf.shape(augment_loss)[0] * tf.shape(augment_loss)[1]]),
        #                            direction='DESCENDING')[0:self.batch_size],
        #                        lambda: tf.reshape(augment_loss,
        #                                           [tf.shape(augment_loss)[0] * tf.shape(augment_loss)[1]]))

        # return loss vector or mean
        bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_augment")
        augment_loss = tf.cond(bool_reduce_mean,
                               lambda: tf.reduce_mean(augment_loss),
                               lambda: augment_loss)

        tf.compat.v1.summary.scalar("self_augmented_loss", tf.reduce_mean(augment_loss))

        return augment_loss, bool_reduce_mean

    @staticmethod
    def _pairwise_distances_from_inputs(in_vec_one, in_vec_two, squared):
        diff_vec = in_vec_one - in_vec_two
        squared_vec = tf.pow(diff_vec, 2)
        distances = tf.reduce_sum(squared_vec, -1)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)
        return distances

    """ -------------------------------- Colour similarity loss -----------------------------------------------------"""

    def colour_similarity_loss_comp_and_incomp(self):
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(squared=False)

        # get cross correlation coefficients
        # in [-1, 1]
        # norm_cross_corr = self._get_normalized_cross_correlations()

        # colour margin
        # in [0,2]
        margin_colour = 1. - self.norm_cross_corr

        # colour loss
        colour_loss = tf.maximum(0., tf.math.abs(pairwise_dist - margin_colour))

        # consider only hardest pairs
        # TODO: Ckeck mining! ...and mean computation
        bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_colour")
        loss_hardest = tf.cond(bool_only_bs_hardest,
                               lambda: tf.sort(
                                   tf.reshape(colour_loss,
                                              [tf.shape(colour_loss)[0] * tf.shape(colour_loss)[1]]),
                                   direction='DESCENDING')[0:self.batch_size],
                               lambda: tf.reshape(colour_loss,
                                                  [tf.shape(colour_loss)[0] * tf.shape(colour_loss)[1]]))

        # return loss vector or mean
        bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_colour")
        colour_loss = tf.cond(bool_reduce_mean,
                              lambda: tf.reduce_mean(loss_hardest),
                              lambda: loss_hardest)

        tf.compat.v1.summary.scalar("hue_saturation_loss", tf.reduce_mean(colour_loss))

        return colour_loss, bool_reduce_mean, bool_only_bs_hardest

    # def _get_normalized_cross_correlations(self):
    #     # shape = [batch_size, 224, 224, 3]
    #     batch_hsv_tensor = tf.image.rgb_to_hsv(self.batch_img_tensor)
    #
    #     # shape = [batch_size, 224, 224, 1]
    #     hue_tensor = batch_hsv_tensor[:, :, :, 0]
    #     saturation_tensor = batch_hsv_tensor[:, :, :, 1]
    #
    #     # # polar coordinates
    #     # # shape = [batch_size, 224, 224, 1]
    #     factor = tf.cast(self.colour_distr_resolution, tf.float32) / 2.
    #     pi_tensor = tf.constant(math.pi)
    #     img_hue_sat_polar_x = factor + saturation_tensor * factor * tf.math.cos(2. * pi_tensor * hue_tensor)
    #     img_hue_sat_polar_y = factor + saturation_tensor * factor * tf.math.sin(2. * pi_tensor * hue_tensor)
    #
    #     # rasterize according to colour_distr_resolution
    #     # -> integer conversion since it was rescaled appropriately
    #     # TODO: Falls _polar == self.colour_distr_resoltuion, dann _polar=_polar-0.x, damit ints nicht out of range bei one_d_indices
    #     # shape = [batch_size, 224, 224, 1]
    #     x_rasterized = tf.cast(img_hue_sat_polar_x, tf.int32)
    #     y_rasterized = tf.cast(img_hue_sat_polar_y, tf.int32)
    #     #
    #     # distribution row-wise
    #     # shape = [batch_size, 224, 224, 1]
    #     # -> contains the indices of a one-d vector assuming that all rows are stacked
    #     one_d_indices = x_rasterized + tf.cast(self.colour_distr_resolution, tf.int32) * y_rasterized
    #     # TODO: minlength expressed via self.colour_distr_resolution
    #     hs_combi_counts = tf.cast(tf.map_fn(lambda cur_ind: tf.math.bincount(cur_ind,
    #                                                                          minlength=25),
    #                                         tf.expand_dims(one_d_indices, -1)), tf.float32)
    #     # mean_hs_count = tf.math.reduce_mean(hs_combi_counts, axis=1)
    #     count_minus_mean = tf.map_fn(
    #         lambda count: count - tf.math.reduce_mean(count),
    #         hs_combi_counts)
    #
    #     # numerator (normalized cross correlation)
    #     mult_of_diff = tf.map_fn(lambda diff_vec: tf.multiply(diff_vec, count_minus_mean), count_minus_mean)
    #     sum_numerat = tf.reduce_sum(mult_of_diff, -1)
    #
    #     # denominator (normalized cross correlation)
    #     c_m_m_square = tf.pow(count_minus_mean, 2)
    #     sum_of_squares = tf.reduce_sum(c_m_m_square, -1)
    #     dot_product_denom = tf.map_fn(lambda square: tf.multiply(square, sum_of_squares), sum_of_squares)
    #     sqrt_denom = tf.sqrt(dot_product_denom)
    #
    #     # main diag == 1
    #     norm_cross_corr = tf.math.divide(sum_numerat, sqrt_denom)
    #
    #     return norm_cross_corr

    """ -------------------------------- Soft triplet loss (new margin) ---------------------------------------------"""

    def soft_triplet_loss_comp_and_incomp(self):
        """
        realises: Triplet loss (Schroff et al., 2015) expanded to
                 - incomplete labels
                 - multiple variables per sample (each with one class label)
                 - optionally multiple labels per variable
        -> margin = Yp(a, p) - Yp(a, n) + 1 - k(a, n) >! 0
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(squared=False)

        x1x2_positive_dist = tf.expand_dims(pairwise_dist, 2)
        x1x3_negative_dist = tf.expand_dims(pairwise_dist, 1)

        # shape(margin) = (batch_size, batch_size, batch_size)
        margin = self._get_incomplete_margin_triplets()

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of x1=i, x2=j, x3=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = x1x2_positive_dist - x1x3_negative_dist + margin

        # Put to zero the invalid triplets
        # where i, j, k == a, p, n not distinct AND
        # where margin(anchor, positive, negative) > 0,
        # i.e. pos.sim(a, p) >= pos.sim(a, n)  + pot.sim(a, n)
        # <=>  yp_apn         >= yp_an         + 1-k_an
        mask = self._get_incomplete_mask_triplets()
        mask = tf.cast(mask, tf.float32)
        triplet_loss_masked_all = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss_masked = tf.maximum(triplet_loss_masked_all, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        # implicitly removes easy triplets by averaging over all hard and
        # semi-hard triplets
        # (already correct distance of triplets' features in feature space)
        valid_triplets = tf.cast(tf.greater(triplet_loss_masked, 1e-16), tf.float32)
        num_positive_triplets = tf.reduce_sum(
            valid_triplets)  # valid triplets in the sense that they are valid and that they are no easy triplets
        num_valid_triplets = tf.reduce_sum(mask)  # valid triplets only in the sense that they are valid
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # batch hard mining: Focus on hardest pairs for training
        # TODO: Ckeck mining! ...and mean computation
        bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_triplet")
        # hardest per anchor
        loss_hardest = tf.cond(bool_only_bs_hardest,
                               lambda: tf.reduce_max(tf.reduce_max(triplet_loss_masked, axis=-1), axis=-1),
                               lambda: tf.reshape(triplet_loss_masked,
                                                  [tf.shape(triplet_loss_masked)[0] *
                                                   tf.shape(triplet_loss_masked)[1] *
                                                   tf.shape(triplet_loss_masked)[2]]))
        # total hardest combinations
        # loss_hardest = tf.cond(bool_only_bs_hardest,
        #                        lambda: tf.sort(
        #                            tf.reshape(triplet_loss_masked,
        #                                       [tf.shape(triplet_loss_masked)[0] * tf.shape(triplet_loss_masked)[1] *
        #                                        tf.shape(triplet_loss_masked)[2]]),
        #                            direction='DESCENDING')[0:self.batch_size],
        #                        lambda: tf.reshape(triplet_loss_masked,
        #                                           [tf.shape(triplet_loss_masked)[0] * tf.shape(triplet_loss_masked)[1] *
        #                                            tf.shape(triplet_loss_masked)[2]]))

        # Get final mean triplet loss over the positive valid triplets
        bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_triplet")
        triplet_loss_final = tf.cond(bool_reduce_mean,
                                     lambda: tf.reduce_sum(triplet_loss_masked)/num_positive_triplets,
                                     lambda: loss_hardest)

        tf.compat.v1.summary.scalar("soft_triplet_loss", tf.reduce_mean(triplet_loss_final))

        return triplet_loss_final, fraction_positive_triplets, bool_reduce_mean, bool_only_bs_hardest

    def _get_incomplete_mask_triplets(self):
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(self.in_label_tensor)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j,
                                                         i_not_equal_k),
                                          j_not_equal_k)

        # check if Yp(a, p) - Yp(a, n) + 1 - k(a, n) >! 0
        margin = self._get_incomplete_margin_triplets()
        x_false = tf.fill(tf.shape(margin), False)
        y_true = tf.fill(tf.shape(margin), True)
        mask_similarity = tf.where(margin <= 0, x_false, y_true)

        # mask
        # shape = (batch_size, batch_size, batch_size)
        mask = tf.logical_and(distinct_indices, mask_similarity)

        return mask

    def _get_incomplete_margin_triplets(self):
        # margin = Yp(a, p) - Yp(a, n) + 1 - k(a, n) >! 0

        # Yp(a, p)
        # shape = (batch_size, batch_size)
        pairwise_similarity, _ = self._get_incomplete_similarity_pairs()
        # shape = (batch_size, batch_size, 1)
        Yp_ap = tf.expand_dims(pairwise_similarity, 2)

        # Yp(a, n)
        # shape = (batch_size, 1, batch_size)
        Yp_an = tf.expand_dims(pairwise_similarity, 1)

        # k(a, n)
        # shape = (batch_size, batch_size)
        pairwise_knowledge = self._get_incomplete_knowledge_pairs()
        # shape = (batch_size, 1, batch_size)
        k_an = tf.expand_dims(pairwise_knowledge, 1)

        # temp = Yp(a, n) + 1 - k(a, n)
        # shape = (batch_size, 1, batch_size)
        temp = Yp_an + 1. - k_an

        # margin
        # shape = (batch_size, batch_size, batch_size)
        # margin = yp_ap - (yp_an + (1-k_an))
        #        = yp_ap - temp
        margin = Yp_ap - temp

        return margin

    def _get_incomplete_knowledge_pairs(self):
        # TODO: was passiert, wenn beide NaN?
        # shape = (batch_size, num_rel_labels)
        if self.multiLabelsListOfVariables is None:
            pi_1 = tf.cast(tf.math.not_equal(self.in_label_tensor, -1), tf.float32)
        else:
            pi_1 = tf.cast(tf.math.not_equal(tf.reduce_sum(self.in_label_tensor, axis=2), 0), tf.float32)

        # shape = (batch_size, 1, num_rel_labels)
        pi_2 = tf.expand_dims(pi_1, 1)

        # shape = (batch_size, batch_size, num_rel_labels)
        pi1_times_pi2 = tf.multiply(pi_1, pi_2)

        # shape = (batch_size, batch_size)
        available_knowledge = tf.math.reduce_sum(tf.multiply(self.label_weight_tensor, pi1_times_pi2), axis=2)

        return available_knowledge

    """ -------------------------------- Soft triplet loss (min margin; Nice 2020) ----------------------------------"""

    def soft_triplet_incomp_loss_min_margin(self):

        # Get the pairwise distance matrix
        pairwise_dist = self._pairwise_distances(squared=False)

        x1x2_positive_dist = tf.expand_dims(pairwise_dist, 2)
        x1x3_negative_dist = tf.expand_dims(pairwise_dist, 1)

        # shape(margin) = (batch_size, batch_size, batch_size)
        margin = self._get_min_margin_from_label_similarity_incomp()

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of x1=i, x2=j, x3=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = x1x2_positive_dist - x1x3_negative_dist + margin

        # Put to zero the invalid triplets
        # where i, j, k not distinct
        # where S(x1,x2) <= S(x1,x3)
        mask = self._get_triplet_min_margin_mask_incomp()
        mask = tf.cast(mask, tf.float32)
        triplet_loss_masked = tf.multiply(mask, triplet_loss)

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss_masked = tf.maximum(triplet_loss, 0.0)

        # Count number of positive triplets (where triplet_loss > 0)
        # implicitly removes easy triplets by averaging over all hard and
        # semi-hard triplets
        # (already correct distance of triplets' features in feature space)
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        num_positive_triplets = tf.reduce_sum(
            valid_triplets)  # valid triplets in the sense that they are valid and that they are no easy triplets
        num_valid_triplets = tf.reduce_sum(mask)  # valid triplets only in the sense that they are valid
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # batch hard mining: Focus on hardest pairs for training
        bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_old_triplet")
        loss_hardest = tf.cond(bool_only_bs_hardest,
                               lambda: tf.sort(
                                   tf.reshape(triplet_loss_masked,
                                              [tf.shape(triplet_loss_masked)[0] * tf.shape(triplet_loss_masked)[1] *
                                               tf.shape(triplet_loss_masked)[2]]),
                                   direction='DESCENDING')[0:self.batch_size],
                               lambda: tf.reshape(triplet_loss_masked,
                                                  [tf.shape(triplet_loss_masked)[0] * tf.shape(triplet_loss_masked)[1] *
                                                   tf.shape(triplet_loss_masked)[2]]))

        # Get final mean triplet loss over the positive valid triplets
        bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_old_triplet")
        triplet_loss = tf.cond(bool_reduce_mean,
                               lambda: tf.reduce_mean(loss_hardest),
                               lambda: loss_hardest)

        tf.compat.v1.summary.scalar("soft_triplet_incomp_loss_min_margin", tf.reduce_mean(triplet_loss))

        return triplet_loss, fraction_positive_triplets, bool_reduce_mean, bool_only_bs_hardest

    def _get_triplet_min_margin_mask_incomp(self):
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(self.in_label_tensor)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j,
                                                         i_not_equal_k),
                                          j_not_equal_k)

        # check if S(x1,x2,L) > S(x,x3,L)
        margin = self._get_min_margin_from_label_similarity_incomp()

        x = tf.fill(tf.shape(margin), False)
        y = tf.fill(tf.shape(margin), True)
        mask_similarity = tf.where(margin <= 0, x, y)

        mask = tf.logical_and(distinct_indices, mask_similarity)

        return mask

    def _get_min_margin_from_label_similarity_incomp(self):

        # shape = (batch_size, 1, num_rel_labels)
        label_acc_x1x3 = tf.expand_dims(self.in_label_tensor, 1)

        # shape = (batch_size, batch_size, num_rel_labels)
        label_equality_p = tf.math.equal(self.in_label_tensor, label_acc_x1x3)
        label_equality_p = tf.cast(label_equality_p, tf.float32)

        label_equality_n = tf.math.not_equal(self.in_label_tensor, label_acc_x1x3)
        label_equality_n = tf.cast(label_equality_n, tf.float32)

        # shape = (batch_size, num_rel_labels)
        label_nan = tf.cast(tf.math.not_equal(self.in_label_tensor, -1), tf.float32)

        # shape = (batch_size, batch_size)
        label_similarity_p = tf.math.reduce_sum(label_equality_p * label_nan, axis=2) / tf.cast(
            tf.shape(self.in_label_tensor)[1],
            tf.float32)
        label_similarity_n = tf.math.reduce_sum(label_equality_n * label_nan, axis=2) / tf.cast(
            tf.shape(self.in_label_tensor)[1],
            tf.float32)

        # shape = (batch_size, batch_size, 1)
        label_similarity_p = tf.expand_dims(label_similarity_p, 2)

        # shape = (batch_size, 1, batch_size)
        label_similarity_n = tf.expand_dims(label_similarity_n, 1)

        # shape = (batch_size, batch_size, batch_size)
        margin = tf.minimum(label_similarity_p, label_similarity_n)

        return margin

    """ -------------------------------- Soft contrastive loss ------------------------------------------------------"""

    def soft_contrastive_loss_comp_and_incomp(self):
        r"""Estimates the constrastive loss for multi-label-based similarity.

        :Arguments\::
            :feature_tensor (*tensor*)\::
                A tf.float tensor with shape = [batch_size, num_features]
                containing the features of the network.
            :in_label_tensor (*tensor*)\::
                Contains the labels of the batch images. Has the shape
                (batch_size, num_labels), where num_labels = len(relevant_variables).
                The entries are the indices of the labels. Belongs to
                feature_tensor.

        :Returns\::
            :loss (*tensor*)\::
                A scalar tf.float Tensor containing the contrastive loss.
            :bool_reduce_mean (*bool*)\::
                Whether to reduce the losses of the batch to one mean loss or not.
        """
        with tf.name_scope("soft_contrastive_incomp_loss"):
            # Epsilon for Norm to avoid NaN in gradients
            eps = tf.constant(1e-15, dtype=tf.float32)

            # Distance d
            # shape (batch_size, batch_size)
            distance = self._pairwise_distances(squared=False)

            # Label similarity Yp, Yn
            # shape (batch_size, batch_size)
            Yp, Yn = self._get_incomplete_similarity_pairs()

            # positive and negative margins Mp, Mn, preliminary loss
            # shape (batch_size, batch_size)
            Mp = self._get_incomplete_pos_margin_pairs()
            Mn = 2. - Mp
            loss_prelim = Yp * tf.maximum(0., distance - Mp) + Yn * tf.maximum(0., Mn - distance)

            # mask invalid pairs
            # invalid: - sample with itself
            #          - Yp = Yn = 0
            mask = self._get_incomplete_mask_pairs(Yp, Yn)
            masked_loss = tf.multiply(mask, loss_prelim)

            # TODO: Ckeck mining! ...and mean computation
            bool_only_bs_hardest = tf.compat.v1.placeholder(tf.bool, name="bool_hard_mining_contrastive")
            # get self.batch_size hardest valid pairs
            # shape = (batch_size * batch_size,)
            loss_hardest = tf.cond(bool_only_bs_hardest,
                                   lambda: tf.sort(
                                       tf.reshape(masked_loss, [tf.shape(masked_loss)[0] * tf.shape(masked_loss)[1]]),
                                       direction='DESCENDING')[0:self.batch_size],
                                   lambda: tf.reshape(masked_loss,
                                                      [tf.shape(masked_loss)[0] * tf.shape(masked_loss)[1]]))

            bool_reduce_mean = tf.compat.v1.placeholder(tf.bool, name="bool_reduce_mean_contrastive")
            loss_reduced = tf.cond(bool_reduce_mean,
                                   lambda: tf.reduce_mean(loss_hardest),
                                   lambda: loss_hardest)

            tf.compat.v1.summary.scalar("soft_contrastive_loss", tf.reduce_mean(loss_reduced))

            return loss_reduced, bool_reduce_mean, bool_only_bs_hardest

    def _get_incomplete_mask_pairs(self, Yp, Yn):
        # 1: valid sample
        # 0: invalid sample

        # check that i, j distinct
        # shape (batch_size, batch_size)
        indices_equal = tf.logical_not(tf.cast(tf.eye(tf.shape(self.in_label_tensor)[0]), tf.bool))

        # check that not (Yp = Yn = 0)
        # shape (batch_size, batch_size)
        yp_yn_is_null = tf.logical_not(tf.logical_and(tf.math.not_equal(Yp, 0),
                                                      tf.math.not_equal(Yn, 0)))

        # mask
        # shape (batch_size, batch_size)
        mask = tf.cast(tf.logical_and(indices_equal, yp_yn_is_null), tf.float32)

        return mask

    def _get_incomplete_pos_margin_pairs(self):

        # shape = (batch_size, num_rel_labels)
        pi_1 = tf.cast(tf.math.not_equal(self.in_label_tensor, -1), tf.float32)

        # shape = (batch_size, 1, num_rel_labels)
        pi_2 = tf.expand_dims(pi_1, 1)

        # shape = (batch_size, batch_size, num_rel_labels)
        pi1_times_pi2 = tf.multiply(pi_1, pi_2)
        one_minus_pipi = 1. - pi1_times_pi2

        # shape = (batch_size, batch_size)
        Mp = tf.math.reduce_sum(one_minus_pipi, axis=2) / tf.cast(tf.shape(self.in_label_tensor)[1],
                                                                  tf.float32)

        return Mp

    def _get_incomplete_similarity_pairs(self):
        r"""Calculates the similarity degree, considering that some labels may be unknown.

        :Arguments\::
            :in_label_tensor (*tensor*)\::
                Contains the label indices of the batch images. Has the shape
                (batch_size, num_lables), where num_labels = len(relevant_variables).

        :Returns\::
            :label_similarity_pos (*tensor*)\::
                A rank-1 tf.float Tensor containing the positive label similarity.
            :label_similarity_neg (*tensor*)\::
                A rank-1 tf.float Tensor containing the negative label similarity.
        """
        if self.multiLabelsListOfVariables is None:
            label_equality_p, label_equality_n = self._get_label_equality_pairs_single_label()
            label_nan = tf.cast(tf.math.not_equal(self.in_label_tensor, -1), tf.float32)
        else:
            label_equality_p, label_equality_n = self._get_label_equality_pairs_multi_label()
            label_nan = tf.cast(tf.math.not_equal(tf.reduce_sum(self.in_label_tensor, axis=2), 0),
                                tf.float32, name="label_nan_float")

        weighted_equality = tf.multiply(self.label_weight_tensor, label_equality_p)
        label_similarity_pos_unreduced = weighted_equality * label_nan
        label_similarity_pos = tf.math.reduce_sum(label_similarity_pos_unreduced, axis=2)
        label_similarity_neg = tf.math.reduce_sum(
            tf.multiply(self.label_weight_tensor, label_equality_n) * label_nan, axis=2)

        return label_similarity_pos, label_similarity_neg

    def _get_label_equality_pairs_multi_label(self):
        mask_false = tf.fill(tf.shape(self.in_label_tensor), False)
        mask_true = tf.fill(tf.shape(self.in_label_tensor), True)
        in_label_tensor_bool_mask = tf.where(tf.equal(self.in_label_tensor, 1), mask_true, mask_false)
        label_acc_x1x3_bool_mask = tf.expand_dims(in_label_tensor_bool_mask, 1)

        label_acc_x1x3 = tf.expand_dims(self.in_label_tensor, 1)
        max_multi_label_divisor = tf.cast(tf.math.maximum(tf.reduce_sum(self.in_label_tensor, axis=2),
                                                          tf.reduce_sum(label_acc_x1x3, axis=3)),
                                          tf.float32)

        label_equality_p_bool_mask = tf.math.logical_and(in_label_tensor_bool_mask,
                                                         label_acc_x1x3_bool_mask)
        label_equality_p_float_mask = tf.cast(label_equality_p_bool_mask, tf.float32)
        label_equality_p_sum = tf.reduce_sum(label_equality_p_float_mask, axis=3)
        label_equality_p_norm = label_equality_p_sum / (max_multi_label_divisor + 1e-16)
        label_equality_p = label_equality_p_norm

        label_equality_n_bool_mask = tf.math.logical_not(label_equality_p_bool_mask)
        label_equality_n_float_mask = tf.cast(label_equality_n_bool_mask, tf.float32)
        label_equality_n_sum = tf.reduce_sum(label_equality_n_float_mask, axis=3)
        label_equality_n_norm = label_equality_n_sum / (max_multi_label_divisor + 1e-16)
        label_equality_n = label_equality_n_norm

        return label_equality_p_norm, label_equality_n

    def _get_label_equality_pairs_single_label(self):
        r"""Calculates the similarity degree, considering that some labels may be unknown.

        :Arguments\::
            :in_label_tensor (*tensor*)\::
                Contains the label indices of the batch images. Has the shape
                (batch_size, num_lables), where num_labels = len(relevant_variables).

        :Returns\::
            :label_similarity_pos (*tensor*)\::
                A rank-1 tf.float Tensor containing the positive label similarity.
            :label_similarity_neg (*tensor*)\::
                A rank-1 tf.float Tensor containing the negative label similarity.
        """
        # Compute label similarity
        # shape = (batch_size, 1, num_rel_labels)
        label_acc_x1x3 = tf.expand_dims(self.in_label_tensor, 1)

        # shape = (batch_size, batch_size, num_rel_labels)
        label_equality_p = tf.math.equal(self.in_label_tensor, label_acc_x1x3)
        label_equality_p = tf.cast(label_equality_p, tf.float32)

        label_equality_n = tf.math.not_equal(self.in_label_tensor, label_acc_x1x3)
        label_equality_n = tf.cast(label_equality_n, tf.float32)

        return label_equality_p, label_equality_n

    def _pairwise_distances(self, squared=False):
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
        dot_product = tf.matmul(self.output_feature_tensor, tf.transpose(self.output_feature_tensor))

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = tf.linalg.tensor_diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances
