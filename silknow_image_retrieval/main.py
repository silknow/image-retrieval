import silknow_image_retrieval as sir

# creates a dataset out of the knowledge graph export
# "collection[1-5].txt" and "collection_rules[1-5].txt" are created
sir.create_dataset_parameter(csvfile=r"./samples/total_post.csv",
                             imgsavepath=r"./samples/data/",
                             master_file_dir=r"./samples/",
                             masterfileRules=r"master_file_rules_dataset_creation.txt")

# performs training on a part of the dataset
# trains on "collection[1-4].txt" and "collection_rules[1-4].txt"
sir.train_model_parameter(master_file_name="master_file_labelled_train.txt",
                          master_file_dir=r"./samples/",
                          master_file_rules_name="master_file_rules_train.txt",
                          master_file_similar="master_file_similar.txt",
                          master_file_dissimilar="master_file_dissimilar.txt",
                          log_dir=r"./output_files/log_dir/",
                          model_dir=r"./output_files/model_dir/")

# build a kD-tree using the trained CNN
# "collection[1-4].txt" and "collection_rules[1-4].txt"  are fed to the tree
sir.build_kDTree_parameter(model_dir=r"./output_files/model_dir/",
                           master_file_tree="master_file_tree.txt",
                           master_dir_tree=r"./samples/",
                           tree_dir=r"./output_files/tree_dir/")

# performs image retrieval for unseen images and kNN classification
# retrives kNN for "collection_5.txt" and "collection_rules_5.txt"
sir.get_kNN_parameter(tree_dir=r"./output_files/tree_dir/",
                      master_file_retrieval="master_file_retrieval.txt",
                      master_dir_retrieval=r"./samples/",
                      model_dir=r"./output_files/model_dir/",
                      pred_gt_dir=r"./output_files/pred_gt_dir/")

# evaluates he kNN classification results
# for the test images in "collection_5.txt" and "collection_rules_5.txt"
sir.evaluate_model_parameter(pred_gt_dir=r"./output_files/pred_gt_dir/",
                             eval_result_dir=r"./output_files/eval_result_dir/")

# performs cross validation
# on all data
sir.cross_validation_parameter(master_file_name="master_file_labelled.txt",
                               master_file_dir=r"./samples/",
                               master_file_rules_name="master_file_rules.txt",
                               master_file_similar="master_file_similar.txt",
                               master_file_dissimilar="master_file_dissimilar.txt",
                               log_dir=r"./output_files/log_dir/",
                               model_dir=r"./output_files/model_dir/",
                               tree_dir=r"./output_files/tree_dir/",
                               pred_gt_dir=r"./output_files/pred_gt_dir/",
                               eval_result_dir=r"./output_files/eval_result_dir/")

