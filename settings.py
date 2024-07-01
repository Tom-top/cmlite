import os
import requests
import platform

import utils.utils as ut

platform_name = platform.system().lower()

###############################################################################
# cmlite paths
###############################################################################

def get_cmlite_path():
    """Returns root path to the ClearMap software

    Returns:
        str: root path to ClearMap
    """
    fn = os.path.split(__file__)
    fn = os.path.abspath(fn[0])
    return fn


cmlite_path = get_cmlite_path()
"""str: Absolute path to the cmlite root folder"""

resources_path = os.path.join(cmlite_path, 'resources')
"""str: Absolute path to the ClearMap resources folder"""

external_path = os.path.join(cmlite_path, 'external')
"""str: Absolute path to the ClearMap external program folder"""

atlas_path = os.path.join(resources_path, "atlas")
config_path = os.path.join(cmlite_path, "config.yml")

###############################################################################
# %% Paths to external programs and resources
###############################################################################

ilastik_path = None
"""str: Absolute path to the Ilastik installation

Notes:
   `Ilastik Webpage <http://ilastik.org/>`_

   `Ilastik Download <http://old.ilastik.org/>`_
"""

# path to eastix installation
elastix_path = os.path.join(external_path, "elastix")
"""str: Absolue path to the elastix installation

Notes:
    `elastix Webpage <http://elastix.isi.uu.nl/>`_
"""

# path to ImageJ/Fiji installation
imagej_path = None
"""str: Absolue path to the ImageJ/Fiji installation

Notes:
    `ImageJ/Fiji Webpage <https://fiji.sc/>`_
"""

# path to TeraSticher installation
terastitcher_path = os.path.join(external_path, 'terastitcher')
"""str: Absolue path to the TeraStitcher installation

Notes:
    `TeraSticher Webpage <http://abria.github.io/TeraStitcher/>`_
"""