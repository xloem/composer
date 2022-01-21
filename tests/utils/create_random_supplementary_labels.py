# Copyright 2021 MosaicML. All Rights Reserved.

import argparse
import csv
import os
import random

from torchvision.datasets.folder import IMG_EXTENSIONS, make_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='Path to dataset. Currently only supports torchvision.datasets.folder.ImageFolder style dataset organization.')
parser.add_argument('--output_directory', type=str, required=True, help='Where to save supplementary label .csv')
parser.add_argument('--output_file', type=str, required=True, help='Name for supplementary label .csv file')

args = parser.parse_args()


def main() -> None:
    samples = make_dataset(args.data_path, extensions=IMG_EXTENSIONS)
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    output_path_full = os.path.join(args.output_directory, args.output_file)
    fieldnames = ["path", "score"]
    with open(output_path_full, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            sample_path = sample[0]
            score = random.uniform(0, 4)
            writer.writerow({"path": sample_path, "score": score})


if __name__ == "__main__":
    main()
