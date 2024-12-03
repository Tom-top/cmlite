import os
import gc
import numpy as np
from natsort import natsorted
import tifffile
from skimage.morphology import skeletonize
from skimage.measure import label

import utils.utils as ut
import resampling.resampling as res

# Initialize the working directory
working_directory = ("/mnt/data/Grace/projectome/fab/080724_rfp_fab_3D_MTG_FULL.dir/"
                     "CTLS Capture - Pos 1 8 [1] 3DMontage Complete-1726216444-719.imgdir")

chunk_directories = [os.path.join(working_directory, i) for i in natsorted(os.listdir(working_directory))
                     if i.startswith("chunk")]
chunk_directories = chunk_directories[3:]
# chunk_directories = [os.path.join(working_directory, "C0")]

for chunk_directory in chunk_directories:
    sample = os.path.basename(chunk_directory)

    skeleton_path = ut.create_dir(os.path.join(chunk_directory, f"skeleton"))

    if not os.path.exists(os.path.join(skeleton_path, f"weighted_skeleton_clean.tif")):

        ################################################################################################################
        # COMBINE THE PREDICTIONS
        ################################################################################################################

        file_paths = natsorted(os.listdir(chunk_directory + "/denardo_1_output_ch0"))
        first_file_path = os.path.join(chunk_directory + "/denardo_1_output_ch0", file_paths[0])
        img_shape = tifffile.imread(first_file_path).shape  # Assuming all images have the same shape
        n_images = len(file_paths)

        # Pre-allocate memory for predictions
        trailmap_prediction = np.zeros((n_images, img_shape[0], img_shape[1]))

        for sm in range(1, 4):
            print(f"Loading prediction Denardo model {sm}")
            trailmap_prediction_path = os.path.join(chunk_directory, f"denardo_{sm}_output_ch0")

            # Process and combine each slice
            for idx, img_name in enumerate(natsorted(os.listdir(trailmap_prediction_path))):
                img = tifffile.imread(os.path.join(trailmap_prediction_path, img_name))

                if sm == 1:
                    img *= 1.34  # Scale only for model 1

                trailmap_prediction[idx] = np.maximum(trailmap_prediction[idx], img)

                del img
                gc.collect()  # Clear memory after processing each image

        gc.collect()  # Final memory clear after processing all subchunks

        ################################################################################################################
        # RESAMPLE THE COMBINED PREDICTION
        ################################################################################################################

        resampled_predicted_path = os.path.join(chunk_directory, "resampled_predicted.tif")
        if not os.path.exists(resampled_predicted_path):
            resample_parameter_predict = {
                "source_resolution": (1, 1, 4),
                "sink_resolution": (5, 5, 5),
                "processes": None,
                "verbose": True,
                "interpolation": None,
            }

            res.resample(np.swapaxes(trailmap_prediction, 0, 2), sink=resampled_predicted_path,
                         **resample_parameter_predict)
            del trailmap_prediction
            gc.collect()  # Clear memory after resampling

            resampled_predicted = tifffile.imread(resampled_predicted_path)
        else:
            print(f"[INFO] Resampling prediction already done: SKIPPING!")

        ################################################################################################################
        # RESAMPLE THE RAW DATA
        ################################################################################################################

        resampled_raw_path = os.path.join(chunk_directory, "resampled_raw.tif")
        input_folder_path = os.path.join(chunk_directory, "input_folder")
        raw_data = [tifffile.imread(os.path.join(input_folder_path, i)) for i in natsorted(os.listdir(input_folder_path))]
        if not os.path.exists(resampled_raw_path):
            resample_parameter_predict = {
                "source_resolution": (1, 1, 4),
                "sink_resolution": (5, 5, 5),
                "processes": None,
                "verbose": True,
                "interpolation": None,
            }

            res.resample(np.swapaxes(raw_data, 0, 2), sink=resampled_raw_path, **resample_parameter_predict)
        else:
            print(f"[INFO] Resampling raw already done: SKIPPING!")

        del raw_data
        gc.collect()  # Clear memory after resampling the raw data

        ################################################################################################################
        # SKELETONIZE THE COMBINED PREDICTION
        ################################################################################################################

        thresholds = np.arange(0.2, 1, 0.1)  # eight separate thresholds
        skeletons = []
        for n, thresh in enumerate(thresholds):
            print(f"Thresholding and skeletonizing the TrailMap prediction: {n + 1}/{len(thresholds)}")
            binarized = resampled_predicted.copy()
            binarized[binarized < thresh] = 0
            binarized[binarized >= thresh] = 1
            skeleton = skeletonize(binarized)

            skeleton_path_tif = os.path.join(skeleton_path, f"skeleton_{str(thresh)[0]}p{str(thresh)[2]}.tif")
            tifffile.imwrite(skeleton_path_tif, skeleton.astype("uint8"))

            del binarized
            skeletons.append(skeleton)
            gc.collect()  # Clear memory after processing each threshold

        weighted_skeletons = [skel * thresh for skel, thresh in zip(skeletons, thresholds)]
        weighted_skeleton = np.sum(weighted_skeletons, axis=0)

        # Normalize and save the final skeleton
        weighted_skeleton_8bit = (((weighted_skeleton - np.min(weighted_skeleton)) * (255 - 0)) /
                                  (np.max(weighted_skeleton) - np.min(weighted_skeleton))) + 0
        tifffile.imwrite(os.path.join(skeleton_path, f"weighted_skeleton.tif"), weighted_skeleton_8bit.astype("uint8"))

        del weighted_skeletons, weighted_skeleton, weighted_skeleton_8bit, resampled_predicted
        gc.collect()  # Clear memory after final skeleton calculation

    else:
        print(f"[INFO] Skeletonization already done: SKIPPING!")
