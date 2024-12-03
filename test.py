import os
import yaml

sample_directory = "/mnt/data/Grace/projectome/1_earmark_gfp_20tiles_2z_3p25zoom_singleill/1_earmark_gfp_20tiles_2z_3p25zoom_singleill.dir/CTLS Capture-1729241760-661.imgdir"
image_record = os.path.join(sample_directory, "ImageRecord.yaml")
with open(image_record, 'r') as yaml_file:
    yaml_data = yaml_file.read()
yaml_documents = yaml_data.strip().split('---')[1].split('StartClass:')
# Initialize a list to store parsed data
# Process each YAML document individually
for doc in yaml_documents:
    doc = "StartClass:" + doc
    # Load the document as YAML
    data = yaml.safe_load(doc)
    # Append to the parsed_data list if valid data is loaded
    try:
        pixel_size = data["StartClass"]["mMicronPerPixel"]
    except:
        pass
    try:
        magnification = data["StartClass"]["mMagnification"]
    except:
        pass

xy_res = pixel_size/magnification
scan_metadata = {}

a = np.load("ImageData_Ch1_TP0000000.npy")
