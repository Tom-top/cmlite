import os
import gc
import importlib

import numpy as np
from natsort import natsorted
import tifffile
from skimage.morphology import skeletonize
from skimage.measure import label

import utils.utils as ut
import resampling.resampling as res

# Define your working directory
working_directory = ("/mnt/data/Grace/projectome/1_earmark_gfp_20tiles_2z_3p25zoom_singleill/raw/sample_1")

chunk_directories = [os.path.join(working_directory, i) for i in natsorted(os.listdir(working_directory))
                     if i.startswith("chunk")]

skeletonize_axons = True
clip_value = 0.9999
weighed_thresh = 70  #20
# weighed_thresh = 0.7
small_object_size = 100

for chunk_directory in chunk_directories:
    print("")
    chunk_name = os.path.basename(chunk_directory)
    ut.print_c(f"[INFO: {chunk_name}] Processing new chunk!")
    sample = os.path.basename(chunk_directory)
    skeleton_path = ut.create_dir(os.path.join(chunk_directory, f"skeleton_2"))
    resampled_chunk_path = os.path.join(chunk_directory, "resampled_raw.tif")

    if not os.path.exists(resampled_chunk_path):
        chunk = tifffile.imread(os.path.join(chunk_directory, "chunk.tif"))
        res.resample(np.swapaxes(chunk, 0, 2), sink=resampled_chunk_path,
                     source_resolution=(2, 2, 2), sink_resolution=(2, 2, 2),
                     processes=None, verbose=True, interpolation=None)

    if not os.path.exists(os.path.join(skeleton_path, f"weighted_skeleton_clean.tif")):
        # Combine the predictions
        file_paths = natsorted(os.listdir(chunk_directory + "/denardo_1_output_ch0"))
        first_file_path = os.path.join(chunk_directory + "/denardo_1_output_ch0", file_paths[0])
        img_shape = tifffile.imread(first_file_path).shape
        n_images = len(file_paths)

        # Resample the combined prediction
        resampled_predicted_path = os.path.join(chunk_directory, "resampled_predicted.tif")

        if not os.path.exists(resampled_predicted_path):
            trailmap_prediction = np.zeros((n_images, img_shape[0], img_shape[1]))

            for sm in range(1, 4):
                ut.print_c(f"[INFO: {chunk_name}] Loading prediction Denardo model {sm}!")
                trailmap_prediction_path = os.path.join(chunk_directory, f"denardo_{sm}_output_ch0")

                for idx, img_name in enumerate(natsorted(os.listdir(trailmap_prediction_path))):
                    img = tifffile.imread(os.path.join(trailmap_prediction_path, img_name))
                    if sm == 1:
                        img *= 1.34
                        img = (img - img.min()) / (img.max() - img.min())
                    img[img >= clip_value] = 0
                    trailmap_prediction[idx] = np.maximum(trailmap_prediction[idx], img)
                    del img
                    gc.collect()

            gc.collect()

            if os.path.exists(resampled_predicted_path):
                os.remove(resampled_predicted_path)
            res.resample(np.swapaxes(trailmap_prediction, 0, 2), sink=resampled_predicted_path,
                         source_resolution=(2, 2, 2), sink_resolution=(2, 2, 2),
                         processes=None, verbose=True, interpolation=None)
            del trailmap_prediction
            gc.collect()
        else:
            ut.print_c(f"[INFO: {chunk_name}] Resampling prediction already done: SKIPPING!")
        resampled_predicted = tifffile.imread(resampled_predicted_path)

        if skeletonize_axons:
            # Skeletonize the combined prediction
            thresholds = np.arange(0.2, 1, 0.1)

            for n, thresh in enumerate(thresholds):
                ut.print_c(f"[INFO: {chunk_name}] Thresholding and skeletonizing the"
                           f" TrailMap prediction: {n + 1}/{len(thresholds)}!")
                binarized = resampled_predicted.copy()
                binarized[binarized < thresh] = 0
                binarized[binarized >= thresh] = 1
                skeleton = skeletonize(binarized)
                tifffile.imwrite(os.path.join(skeleton_path, f"skeleton_{str(thresh)[0]}p{str(thresh)[2]}.tif"),
                                 skeleton.astype("uint8"))
                del binarized, skeleton
                gc.collect()

            del resampled_predicted
            gc.collect()

            # Accumulate weighted skeletons
            weighted_skeletons_sum = None
            ut.print_c(f"[INFO: {chunk_name}] Generating weighed skeleton!")

            for thresh in thresholds:
                ut.print_c(f"[INFO: {chunk_name}] Processing threshold {thresh}")
                skeleton_file = os.path.join(skeleton_path, f"skeleton_{str(thresh)[0]}p{str(thresh)[2]}.tif")
                skeleton = tifffile.imread(skeleton_file)
                skeleton_weighted = skeleton * thresh

                if weighted_skeletons_sum is None:
                    weighted_skeletons_sum = skeleton_weighted
                else:
                    weighted_skeletons_sum += skeleton_weighted

                del skeleton, skeleton_weighted
                gc.collect()

            weighted_skeleton_8bit = (((weighted_skeletons_sum - np.min(weighted_skeletons_sum)) * 255) /
                                      (np.max(weighted_skeletons_sum) - np.min(weighted_skeletons_sum)))
            tifffile.imwrite(os.path.join(skeleton_path, f"weighted_skeleton.tif"), weighted_skeleton_8bit.astype("uint8"))

            skeletonized_axons_bin = (weighted_skeleton_8bit > weighed_thresh).astype("uint8")
            tifffile.imwrite(os.path.join(skeleton_path, f"binarized_skeleton.tif"), skeletonized_axons_bin)
            del weighted_skeletons_sum, weighted_skeleton_8bit
            gc.collect()
        else:
            skeletonized_axons_bin = (resampled_predicted > weighed_thresh).astype("uint8")
            tifffile.imwrite(os.path.join(skeleton_path, f"binarized_prediction.tif"), skeletonized_axons_bin)

        # Cleaning small components in skeleton
        ut.print_c(f"[INFO: {chunk_name}] Cleaning up combined skeleton!")
        labels, num_features = label(skeletonized_axons_bin, return_num=True, connectivity=3)

        component_sizes = np.bincount(labels.ravel())
        small_labels = np.where(component_sizes < small_object_size)[0]
        small_components = np.isin(labels, small_labels)
        cleaned_skeleton = (skeletonized_axons_bin > 0) & ~small_components

        tifffile.imwrite(os.path.join(skeleton_path, f"binarized_skeleton_clean.tif"),
                         cleaned_skeleton.astype('uint8'))

        if skeletonize_axons:
            masked_weighted_skeleton_8bit = tifffile.imread(os.path.join(skeleton_path, f"weighted_skeleton.tif"))
            cleaned_skeleton = tifffile.imread(os.path.join(skeleton_path, f"binarized_skeleton_clean.tif"))
            masked_weighted_skeleton_8bit[~(cleaned_skeleton == 1)] = 0
            tifffile.imwrite(os.path.join(skeleton_path, f"weighted_skeleton_clean.tif"), masked_weighted_skeleton_8bit)
            del masked_weighted_skeleton_8bit
        del component_sizes, small_labels, small_components, cleaned_skeleton
        gc.collect()

        # Clear numpy array cache
        np.lib.format.open_memmap = None
        gc.collect()

        # Reload modules
        importlib.reload(ut)
        importlib.reload(res)

    else:
        ut.print_c(f"[INFO: {chunk_name}] Skeletonization already done: SKIPPING!")

    # Clean up and ensure memory reset at the end of each iteration
    del chunk_directory, sample, skeleton_path
    gc.collect()
