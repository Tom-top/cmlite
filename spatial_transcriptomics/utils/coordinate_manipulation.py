import time

import numpy as np

def filter_points_in_3d_mask(arr_0, mask_1):
    start_time = time.time()
    int_coords = arr_0.astype(int)
    valid_x = (0 <= int_coords[:, 0]) & (int_coords[:, 0] < mask_1.shape[0])
    valid_y = (0 <= int_coords[:, 1]) & (int_coords[:, 1] < mask_1.shape[1])
    valid_z = (0 <= int_coords[:, 2]) & (int_coords[:, 2] < mask_1.shape[2])
    in_bounds = valid_x & valid_y & valid_z

    mask_2 = np.zeros(arr_0.shape[0], dtype=bool)
    mask_2[in_bounds] = mask_1[int_coords[in_bounds, 0], int_coords[in_bounds, 1], int_coords[in_bounds, 2]] == 255
    filtered_arr_0 = arr_0[mask_2]

    end_time = time.time()
    print(f"Function run time: {end_time - start_time} seconds")

    return filtered_arr_0, mask_2
