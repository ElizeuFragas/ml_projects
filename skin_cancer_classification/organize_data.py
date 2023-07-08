"""
Code to organize data from
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
into three distinct CSV files, with train, test and validation sets.

Assuming this script is executed at folder CODE_FOLDER, put the
data  in a parent parant folder called ../../data_ham1000.

Assume that all JPG images were manually copied to the same folder.
For instance, I moved the files in the original folder
HAM10000_images_part_2 into
HAM10000_images_part_1, such that all JPG files are into a unique folder
(../../data_ham1000/HAM10000_images_part_1)
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_according_to_lesion(df, test_size_fraction=0.2):
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Group the dataframe by 'lesion_id'
    grouped = df.groupby("lesion_id")["image_name"].apply(list).reset_index()

    # Split the grouped data into training and test sets
    train_df, test_df = train_test_split(
        grouped, test_size=test_size_fraction, random_state=42
    )

    # Merge the training and test sets back with the original dataframe
    train_merged = pd.merge(df, train_df, on="lesion_id")
    test_merged = pd.merge(df, test_df, on="lesion_id")

    # remove undesired columns created by pd.merge
    train_merged.pop("image_name_y")
    test_merged.pop("image_name_y")

    # rename columns
    train_merged.rename(columns={"image_name_x": "image_name"}, inplace=True)
    test_merged.rename(columns={"image_name_x": "image_name"}, inplace=True)

    return train_merged, test_merged


all_ham_data_df = pd.read_csv("../../data_ham1000/HAM10000_metadata.csv")

print(all_ham_data_df.info())
print(all_ham_data_df.head())

dx = all_ham_data_df["dx"]
print(np.unique(dx))

# original labels: ['akiec' 'bcc' 'bkl' 'df' 'mel' 'nv' 'vasc']
positive_labels = ["akiec", "bcc", "bkl", "df", "mel", "vasc"]
negative_label = "nv"

# neg_examples = ham_data[ham_data['dx'] == negative_labels].copy()
# print(neg_examples.info())
all_ham_data_df.loc[all_ham_data_df["dx"] == negative_label, "dx"] = "0"

# https://www.askpython.com/python-modules/pandas/update-the-value-of-a-row-dataframe
for i in range(len(positive_labels)):
    this_positive_label = positive_labels[i]
    all_ham_data_df.loc[all_ham_data_df["dx"] == this_positive_label, "dx"] = "1"

dx = all_ham_data_df["dx"]
print(np.unique(dx))

# rename columns
all_ham_data_df.rename(columns={"dx": "target"}, inplace=True)
all_ham_data_df.rename(columns={"image_id": "image_name"}, inplace=True)

# Add the extension '.jpg' to all items in the 'image_name' column
all_ham_data_df["image_name"] = all_ham_data_df["image_name"].apply(
    lambda x: x + ".jpg"
)


print(all_ham_data_df.info())
print(all_ham_data_df.head())

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
output_file_name = "../../data_ham1000/binary_HAM10000_metadata.csv"
all_ham_data_df.to_csv(output_file_name)
print("Wrote", output_file_name)

train_df, test_df = split_according_to_lesion(all_ham_data_df, test_size_fraction=0.3)
train_df, validation_df = split_according_to_lesion(train_df, test_size_fraction=0.2)

if True:
    print("train:")
    print(train_df["target"].value_counts())
    print("test:")
    print(test_df["target"].value_counts())
    print("validation:")
    print(validation_df["target"].value_counts())

output_file_name = "../../data_ham1000/train.csv"
train_df.to_csv(output_file_name)
print("Wrote", output_file_name)

output_file_name = "../../data_ham1000/test.csv"
test_df.to_csv(output_file_name)
print("Wrote", output_file_name)

output_file_name = "../../data_ham1000/validation.csv"
validation_df.to_csv(output_file_name)
print("Wrote", output_file_name)
