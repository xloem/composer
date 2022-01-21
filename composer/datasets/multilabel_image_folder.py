# Copyright 2021 MosaicML. All Rights Reserved.

import ast
import csv
import logging
from typing import Any, Callable, Dict, Optional

import torch
from torchvision.datasets.folder import ImageFolder, default_loader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class MultilabelImageFolder(ImageFolder):
    """Inherits :class:`~torchvision.datasets.ImageFolder`and adds support for
    supplementary sample labels. __getitem__() returns a dict instead of a tuple.
    
    The supplementary labels are contained in a .csv. Each row of the supplementary sample
    label .csv is expected to be of the form:
    `path,label_1,label_2,...label_n`
    `{path_to_sample},{value_0},{value_1}...{label_n}`,...
    where the first row is the column labels

    A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        supp_label_path (string, optional): Path to .csv containing supplementary sample
            labels.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 supp_label_path: Optional[str] = None):
        super(MultilabelImageFolder, self).__init__(root,
                                                    transform=transform,
                                                    target_transform=target_transform,
                                                    loader=loader,
                                                    is_valid_file=is_valid_file)

        parsed_rows = {}
        if not isinstance(supp_label_path, type(None)):
            # Parse .csv into a dict in which each item corresponds to a row, and each row
            # is encoded as a dict. The data structure takes the following form:
            # {
            # sample0_path: {label0: value0, label1: value1, ...},
            # sample1_path: {label0: value0, label1: value1, ...},
            # ...
            # sampleN_path: {label0: value0, label1: value1, ...}
            # }
            try:
                with open(supp_label_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row_n, row in enumerate(reader, start=1):
                        # parsed_row = {}
                        sample_path = None
                        for k, v in row.items():
                            # Default values
                            parsed_val = float("nan")
                            if k == "path":
                                parsed_rows[v] = {}
                                sample_path = v
                            else:
                                try:
                                    parsed_val = ast.literal_eval(v)
                                except ValueError:
                                    log.warning(
                                        f"Unable to parse column value {v} from row {row_n} in supplementary label file {supp_label_path}. Strings must be 'quoted' to be parsed correctly."
                                    )
                                try:
                                    parsed_rows[sample_path][k] = parsed_val
                                except KeyError:
                                    log.warning(
                                        f"Unable to parse sample path from row {row_n} in supplementary label file {supp_label_path}. Please ensure the first column is labeled 'path'."
                                    )
                        # rows.append(parsed_row)
            except FileNotFoundError:
                log.error(f"Supplementary label file {supp_label_path} not found.")

        # Turn self.samples from list of tuples into list of dicts and add
        # supplementary labels.
        sample_dicts = []
        n_failed_supp_labels = 0
        for sample in self.samples:
            sample_path, target = sample
            curr_sample_dict = {"path": sample_path, "target": target}
            if parsed_rows:
                try:
                    curr_row = parsed_rows[sample_path]
                    for k, v in curr_row.items():
                        curr_sample_dict[k] = v
                except KeyError:
                    n_failed_supp_labels += 1
            sample_dicts.append(curr_sample_dict)
        self.samples = sample_dicts
        if n_failed_supp_labels > 0:
            log.warning(f"Unable to add one or more supplementary labels to {n_failed_supp_labels} samples.")

    def __getitem__(self, index: int) -> Dict:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample_dict = self.samples[index]
        # Do we need to worry about dtype?
        return_dict = {k: torch.tensor(v) for k, v in sample_dict.items() if k not in ["path", "target"]}
        sample = self.loader(sample_dict["path"])
        target = sample_dict["target"]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return_dict["data"] = sample
        return_dict["target"] = target
        return return_dict
