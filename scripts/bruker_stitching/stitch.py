import os

import stitching.stitching as st

raw_directory = r"/default/path"  # PERSONAL

parameters = {"study_params":
                  {"scanning_system": "",
                   "samples_to_process": [],
                   "channels_to_stitch": [3]},
              "stitching": {"search_params": [20, 20, 5],
                            "z_subreg_alignment": [550, 650],
                            },
              }

st.stitch_samples(raw_directory, **parameters)
