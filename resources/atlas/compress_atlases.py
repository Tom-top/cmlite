import os
import lzma
import tifffile

def compress_tif_to_lzma(input_file, output_file):
    # Read the TIFF image as bytes
    with open(input_file, "rb") as file:
        tif_data = file.read()

    # Compress the byte data using LZMA
    compressed_data = lzma.compress(tif_data)

    # Write the compressed data to the output file
    with open(output_file, "wb") as file:
        file.write(compressed_data)

    print(f"File {input_file} compressed to {output_file}")

atlas_path = "resources/atlas"
input_files = [i for i in os.listdir(atlas_path) if i.endswith("tif")]

for input_file in input_files:
    compress_tif_to_lzma(os.path.join(atlas_path, input_file),
                         os.path.join(atlas_path, input_file + ".lzma"))