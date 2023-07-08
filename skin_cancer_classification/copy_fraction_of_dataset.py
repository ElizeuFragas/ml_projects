"""
This script copies part of the original dataset such
that a smaller version can be used by people that does
not want to download the complete 6 GB dataset. It also
writes a CSV file with all files in the reduced version
of the dataset.
"""


import os
import shutil

import pandas as pd

# Fraction of files to copy (e.g., 0.5 for 50% of files)
fraction_to_copy = 0.08

df = pd.read_csv("../../data_ham1000/HAM10000_metadata.csv")

# Destination folder
destination_folder = "../../data_ham1000_small/"

# Calculate the number of files to copy based on the fraction
num_files = int(len(df) * fraction_to_copy)

# Randomly select 'num_files' files from the DataFrame
files_to_copy = df.sample(n=num_files, random_state=42)

# Iterate over the selected files and copy them to the destination folder
i = 0
for file_name in files_to_copy["image_id"]:
    source_file = os.path.join(
        "D:/github/data_ham1000/HAM10000_images_part_1", file_name + ".jpg"
    )
    destination_file = os.path.join(destination_folder, file_name + ".jpg")
    print(i, destination_file)
    shutil.copyfile(source_file, destination_file)
    i += 1

# Save the DataFrame to a CSV file
files_to_copy.to_csv(destination_folder + "HAM10000_metadata_small.csv", index=False)
