import os

import utils.utils as ut

import alignment.align as elx

atlas = "gubra"
auto_channel = 1
signal_channel = 3
auto_directory = fr"/default/path"  # PERSONAL
signal_directory = fr"/default/path"  # PERSONAL
reference_10um_path = r"C:\Users\MANDOUDNA\PycharmProjects\cmlite\resources\atlas\gubra_reference_mouse_10um_3_2_1_None.tif"

for s in os.listdir(auto_directory):
    if s.endswith(".tif"):
        s_name = s.split(".")[0]
        auto_path = os.path.join(auto_directory, s)
        signal_path = os.path.join(signal_directory, s)

        ########################################################################################################
        # 2.6.1 ALIGN AUTO TO REFERENCE (ABA)
        ########################################################################################################

        auto_to_reference_aba_directory = os.path.join(auto_directory,
                                                       f"{s_name}_{atlas}_auto_to_reference_10um_{auto_channel}")
        align_auto_to_reference_10um = dict(fixed_image_path=reference_10um_path,
                                            moving_image_path=auto_path,
                                            affine_parameter_file="resources/alignment/align_affine_10um.txt",
                                            bspline_parameter_file="resources/alignment/align_bspline.txt",
                                            output_dir=auto_to_reference_aba_directory,
                                            )

        if not os.path.exists(auto_to_reference_aba_directory):
            ut.print_c(
                f"[INFO {s_name}] Running auto to {atlas} reference (10um) alignment for channel {auto_channel}!")
            elx.align_images(**align_auto_to_reference_10um)
        else:
            ut.print_c(
                f"[WARNING {s_name}] Alignment: auto to {atlas} reference (10um) skipped for channel {auto_channel}: "
                f"auto_to_reference_10um_{auto_channel} folder already exists!")

        ########################################################################################################
        # 2.6.2 TRANSFORM SIGNAL TO REFERENCE
        ########################################################################################################

        signal_to_reference_10um_directory = os.path.join(signal_directory,
                                                          f"{s_name}_{atlas}_signal_to_reference_10um_{signal_channel}")
        transform_atlas_parameter = dict(
            source=signal_path,
            result_directory=signal_to_reference_10um_directory,
            transform_parameter_file=os.path.join(auto_to_reference_aba_directory,
                                                  f"TransformParameters.1.txt"))
        if not os.path.exists(signal_to_reference_10um_directory):
            ut.print_c(
                f"[INFO {s_name}] Running signal to {atlas} reference (10um) transform for channel {signal_channel}!")
            elx.transform_images(**transform_atlas_parameter)
        else:
            ut.print_c(
                f"[WARNING {s_name}] Transforming: signal to {atlas} reference (10um) skipped for channel {signal_channel}: "
                f"signal_to_reference_10um_{signal_channel} folder already exists!")
