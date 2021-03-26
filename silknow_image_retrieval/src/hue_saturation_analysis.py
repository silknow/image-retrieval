# -*- coding: utf-8 -*-
"""
Created on Wed July 15 11:43:10 2020

@author: dorozynski
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from shutil import copy
from skimage.color import rgb2hsv
from imageio import imread
from math import cos, sin, pi
from sklearn.cluster import KMeans
from sklearn import preprocessing

# sys.path.insert(0, '../')
# import SILKNOW_WP4_library as sn_func

from . import SILKNOW_WP4_library as wp4lib

def get_hue_and_saturation_image(im_path, im_name):
    """Loads and image and transforms it to HSV.

    :Arguments\::
        :im_path (*string*)\::
            The directory where the master file is stored.
        :im_name (*string*)\::
            The name of the master file.

    :Returns\::
        :img_hsv (*image*)\::
            An image in HSV colour space produced by skimage.color.rgb2hsv.
        :img_hue (*image*)\::
            The hue channel (values in [0, 1]) of the image in img_hsv.
        :img_sat (*image*)\::
            The saturation channel (values in [0, 1]) of the image in img_hsv.
    """
    img_rgb = imread(os.path.join(im_path, im_name))
    img_hsv = rgb2hsv(img_rgb)
    img_hue = img_hsv[:, :, 0]
    img_sat = img_hsv[:, :, 1]

    #     fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
    #     ax0.imshow(img_rgb)
    #     ax0.set_title("RGB image")
    #     ax0.axis('off')
    #     ax1.imshow(img_hue, cmap='hsv')
    #     ax1.set_title("Hue channel")
    #     ax1.axis('off')
    #     ax2.imshow(img_sat)
    #     ax2.set_title("Saturation channel")
    #     ax2.axis('off')
    #     fig.tight_layout()

    return img_hsv, img_hue, img_sat


def get_polar_coordinates(h, s, m):
    """Loads and image and transforms it to HSV.

        :Arguments\::
            :h (*image*)\::
                The hue channel (values in [0, 1]) of an HSV image produced by skimage.color.rgb2hsv.
            :s (*image*)\::
                The hue saturation (values in [0, 1]) of an HSV image produced by skimage.color.rgb2hsv.
            :m (*int*)\::
                A margin defining the function values of the polar coordinates.
                The values will be in [0, m] in or on a circle with centre [m/2, m/2] and radius m/2.

        :Returns\::
            :x (*float*)\::
                Horizontal polar coordinate.
            :y (*float*)\::
                Vertical polar coordinate.
        """
    x = m / 2 + s * m / 2 * cos(h * 2 * pi)
    y = m / 2 + s * m / 2 * sin(h * 2 * pi)

    return (x, y)


v_get_polar_coordinates = np.vectorize(get_polar_coordinates)


def get_polar_counter_matrix_from_img_file(im_path, im_name, resolution):
    # hue and saturation will be in the range [0, 1]
    img_hsv, img_hue, img_sat = get_hue_and_saturation_image(im_path, im_name)

    arr_hue = np.array(img_hue)
    arr_sat = np.array(img_sat)

    (img_hue_sat_polar_x, img_hue_sat_polar_y) = v_get_polar_coordinates(arr_hue, arr_sat, resolution)

    img_hue_sat_polar_x[np.where(img_hue_sat_polar_x==resolution)[0][:]] -= 0.1
    img_hue_sat_polar_y[np.where(img_hue_sat_polar_y==resolution)[0][:]] -= 0.1

    interval = np.linspace(0, resolution, resolution + 1)
    interval[-1] += 1e-10
    hue_sat_counter = np.zeros((resolution, resolution))

    last_y_bin = 0
    for y_ind, y_bin in enumerate(interval[1::]):
        y_in_cur_bin = (last_y_bin <= img_hue_sat_polar_y) & (img_hue_sat_polar_y < y_bin)  # boolean image array
        last_y_bin = y_bin
        last_x_bin = 0
        for x_ind, x_bin in enumerate(interval[1::]):
            x_in_cur_bin = (last_x_bin <= img_hue_sat_polar_x) & (img_hue_sat_polar_x < x_bin)  # boolean image array
            cur_count = np.sum((y_in_cur_bin & x_in_cur_bin).astype(int))
            last_x_bin = x_bin

            hue_sat_counter[y_ind, x_ind] = cur_count

    # y_hist = np.histogram(img_hue_sat_polar_y, bins=interval)
    # x_hist = np.histogram(img_hue_sat_polar_x, bins=interval)
    #
    # fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 3))
    # ax0.imshow(hue_sat_counter)
    # ax0.set_title("hue_sat_counter")
    # ax0.set_xlabel("counts s*cos(h*2*pi)")
    # ax0.set_ylabel("counts s*sin(h*2*pi)")
    #
    # plt.sca(ax1)
    # plt.hist(interval[1::], weights=y_hist[0])
    # ax1.set_title("y hist")
    #
    # plt.sca(ax2)
    # plt.hist(interval[1::], weights=x_hist[0])
    # ax2.set_title("x hist")
    #
    # fig.tight_layout()

    return hue_sat_counter


def get_polar_counter_from_img_file(im_path, im_name, resolution):
    # hue and saturation will be in the range [0, 1]
    img_hsv, img_hue, img_sat = get_hue_and_saturation_image(im_path, im_name)

    arr_hue = np.array(img_hue)
    arr_sat = np.array(img_sat)

    (img_hue_sat_polar_x, img_hue_sat_polar_y) = v_get_polar_coordinates(arr_hue, arr_sat, resolution)

    img_hue_sat_polar_x[np.where(img_hue_sat_polar_x==resolution)[0][:]] -= 0.1
    img_hue_sat_polar_y[np.where(img_hue_sat_polar_y==resolution)[0][:]] -= 0.1

    one_d_indices = img_hue_sat_polar_x.astype(int) + resolution * img_hue_sat_polar_y.astype(int)
    one_d_indices = np.reshape(one_d_indices, (np.shape(one_d_indices)[0]*np.shape(one_d_indices)[1]))

    hue_sat_counter = np.bincount(one_d_indices, minlength=25)

    return hue_sat_counter


def get_polar_counter_from_img_arr(img_arr, resolution):
    # hue and saturation will be in the range [0, 1]
    img_hsv = rgb2hsv(img_arr)
    img_hue = img_hsv[:, :, 0]
    img_sat = img_hsv[:, :, 1]

    arr_hue = np.array(img_hue)
    arr_sat = np.array(img_sat)

    (img_hue_sat_polar_x, img_hue_sat_polar_y) = v_get_polar_coordinates(arr_hue, arr_sat, resolution)

    img_hue_sat_polar_x[np.where(img_hue_sat_polar_x==resolution)[0][:]] -= 0.1
    img_hue_sat_polar_y[np.where(img_hue_sat_polar_y==resolution)[0][:]] -= 0.1

    one_d_indices = img_hue_sat_polar_x.astype(int) + resolution * img_hue_sat_polar_y.astype(int)
    one_d_indices = np.reshape(one_d_indices, (np.shape(one_d_indices)[0]*np.shape(one_d_indices)[1]))

    hue_sat_counter = np.bincount(one_d_indices, minlength=25)

    return hue_sat_counter


def get_polar_counter_matrices_for_batch(batch_img_arr, resolution):
    batch_hs_counter = []
    for img_arr in tqdm(batch_img_arr):
        cur_hs_counter = get_polar_counter_from_img_arr(img_arr, resolution)
        batch_hs_counter.append(cur_hs_counter)
    return batch_hs_counter


def get_normalized_cross_correlation_batch(batch_input):
    batch_input = np.asarray(batch_input).astype(float)
    count_minus_mean = np.apply_along_axis(lambda count: count-np.mean(count), 0, batch_input)

    mult_of_diff = [np.multiply(diff_vec, count_minus_mean) for diff_vec in count_minus_mean]
    sum_numerat = np.sum(mult_of_diff, -1)

    c_m_m_square = np.power(count_minus_mean, 2)
    sum_of_squares = np.sum(c_m_m_square, -1)
    dot_product_denom = [np.multiply(square, sum_of_squares) for square in sum_of_squares]

    sqrt_denom = np.sqrt(dot_product_denom)

    norm_cross_corr = np.divide(sum_numerat, sqrt_denom)

    return norm_cross_corr


def get_descriptor_from_HS_matrix(HS_matrix):
    return np.reshape(HS_matrix, np.size(HS_matrix))


def get_all_colour_descriptors(im_fullname_list, resolution):
    colour_descr = []
    for fullname in im_fullname_list:
        im_path, im_name = os.path.split(fullname)

        cur_descr_mat = get_polar_counter_matrix_from_img_file(im_path, im_name, resolution)
        cur_descr_vek = get_descriptor_from_HS_matrix(cur_descr_mat)

        colour_descr.append(cur_descr_vek)
    return colour_descr


def get_image_names_from_masterfile(masterfile_dir, masterfile_name):
    coll_list = sn_func.master_file_to_collections_list(masterfile_dir, masterfile_name)
    im_fullname_list = []
    for collection in coll_list:
        coll_id = open(os.path.join(masterfile_dir, collection).strip("\n"), 'r')
        for line, rel_im_path in enumerate(coll_id):
            if line == 0: continue
            image_name = os.path.abspath(os.path.join(masterfile_dir,
                                                      rel_im_path.split('\t')[0].strip("\n")))
            im_fullname_list.append(image_name)
        coll_id.close()
    return im_fullname_list


def get_k_means_cluster_indices(all_descr, num_clusters):
    # runs 10 times k-means and takes the best result
    # each run performs 300 iterations or stops if the cluster centers move less that 1e-4 between two consecutive iterations (Frobenius norm)
    # no fixed random seed for centers
    # kmeans.inertia_: Sum of squared distances of samples to their closest cluster center.
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descr)
    cluster_labels = kmeans.predict(all_descr)

    return cluster_labels


def write_clustering_result(im_fullname_list, result_path, cluster_labels, num_clusters, descr_resolution):
    cluster_stat_df = pd.DataFrame({"cluster_indice": [], "full_im_name": [], "SILKNOW_obj_URI": []})
    for cluster in range(num_clusters):
        ind_images_cluster = np.where(cluster_labels == cluster)
        full_im_name = [im_fullname_list[i] for i in ind_images_cluster[0]]
        SILKNOW_obj_URI = [os.path.split(im_fullname_list[i])[1].split("__")[1] for i in ind_images_cluster[0]]
        cluster_indice = cluster * np.ones(len(ind_images_cluster[0]))

        cur_df = pd.DataFrame(
            {"cluster_indice": cluster_indice, "full_im_name": full_im_name, "SILKNOW_obj_URI": SILKNOW_obj_URI})
        cluster_stat_df = cluster_stat_df.append(cur_df, ignore_index=True)

    main_dir = os.path.join(result_path, "clustering_result_" + str(int(num_clusters)) + "_clusters_" + str(
        int(descr_resolution)) + "_descriptor_resolution")
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    # save dataframe as CSV
    cluster_stat_df.to_csv(os.path.join(main_dir, "colour_clustering_result.csv"), index=False)

    for cluster in np.unique(cluster_stat_df.cluster_indice):
        cluster_dir = os.path.join(main_dir, "images_in_cluster_" + str(int(cluster)))

        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        for full_image_name in cluster_stat_df[cluster_stat_df.cluster_indice == cluster].full_im_name:
            copy(full_image_name, os.path.join(cluster_dir, os.path.split(full_image_name)[1]))


def save_descriptors(result_path, descr_resolution, all_descr):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    np.savez(result_path + r"/descriptor_resolution_" + str(int(descr_resolution)) + ".npz", all_descr)


def create_colour_descriptor_and_perform_clustering(masterfile_dir, masterfile_name, descr_resolution, num_clusters,
                                                    result_path):
    im_fullname_list = get_image_names_from_masterfile(masterfile_dir, masterfile_name)
    all_descr = get_all_colour_descriptors(im_fullname_list, descr_resolution)

    # scale to zero mean and unit variance
    all_descr_sclaed = preprocessing.scale(all_descr)

    cluster_labels = get_k_means_cluster_indices(all_descr_sclaed, num_clusters)
    write_clustering_result(im_fullname_list, result_path, cluster_labels, num_clusters, descr_resolution)


def clustering_from_descriptors(descriptors, num_clusters, masterfile_dir, masterfile_name, result_path):
    im_fullname_list = get_image_names_from_masterfile(masterfile_dir, masterfile_name)
    descr_resolution = int(os.path.split(descriptors)[1].split(".")[0].split("_")[2])

    all_descr = np.load(descriptors)["arr_0"]

    # scale to zero mean and unit variance
    all_descr_sclaed = preprocessing.scale(all_descr)

    cluster_labels = get_k_means_cluster_indices(all_descr_sclaed, num_clusters)
    write_clustering_result(im_fullname_list, result_path, cluster_labels, num_clusters, descr_resolution)
