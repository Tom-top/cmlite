import os
import requests

import utils.utils as ut


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

atlas_files_link = ("https://www.dropbox.com/scl/fo/ld9re7620kela1oovblsj/ADDly_yw2M0huvf-bS1DsfE?rlkey=5ctl8a5sdc882nu8"
                    "hesrx9ca9&st=vj4nwexz&dl=0")

# def download_file_from_cloud(url, output_path):
#     if not os.path.exists(output_path):
#         # Ensure the link is a direct download link
#         if '?dl=0' in url:
#             url = url.replace('?dl=0', '?dl=1')
#         elif '?dl=1' not in url:
#             url += '?dl=1'
#
#         try:
#             # Send the GET request to download the file
#             response = requests.get(url, stream=True)
#             response.raise_for_status()  # Ensure we notice bad responses
#
#             # Check the content type to ensure it's not an HTML error page
#             content_type = response.headers.get('Content-Type', '')
#             if 'text/html' in content_type:
#                 print("Error: The response content is an HTML page. Please check the Dropbox link.")
#                 return
#
#             # Check for actual content disposition to ensure file download
#             content_disposition = response.headers.get('Content-Disposition', '')
#             if not content_disposition:
#                 print("Error: Content-Disposition header not found. Please check the Dropbox link.")
#                 return
#
#             # Open a local file with write-binary mode
#             with open(output_path, 'wb') as file:
#                 # Write the content of the response to the file
#                 for chunk in response.iter_content(chunk_size=8192):
#                     file.write(chunk)
#             print(f"File downloaded successfully and saved to {output_path}")
#
#         except requests.exceptions.RequestException as e:
#             print(f"Failed to download the file. Error: {e}")


cmlite_path = get_cmlite_path()
"""str: Absolute path to the cmlite root folder"""

resources_path = os.path.join(cmlite_path, 'resources')
"""str: Absolute path to the ClearMap resources folder"""

external_path = os.path.join(cmlite_path, 'external')
"""str: Absolute path to the ClearMap external program folder"""

atlas_path = os.path.join(resources_path, "atlas")

# aba_annotation_mouse = ("https://www.dropbox.com/scl/fi/1p5gagdxj5l1izb8tgcc5/ABA_annotation_mouse.tif.lzma?rlkey=o2pf9n"
#                         "0qcniteqyunka54vmgm&st=zm8nl5lx&dl=1")
# aba_reference_mouse = ("https://www.dropbox.com/scl/fi/9f7g887ooh1jgru9in9dz/ABA_reference_mouse.tif.lzma?rlkey=u9hnkq5"
#                        "sonpoh72infekgb1cz&st=ftm0vuoq&dl=1")
#
# download_file_from_cloud(aba_annotation_mouse, os.path.join(atlas_path, "ABA_annotation_mouse.tif.lzma"))
# download_file_from_cloud(aba_reference_mouse, os.path.join(atlas_path, "ABA_reference_mouse.tif.lzma"))

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