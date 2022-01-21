# Copyright 2021 MosaicML. All Rights Reserved.

import logging
from typing import Any, Callable, Dict, Optional

import torch
from torchvision.datasets.folder import ImageFolder, default_loader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class ImageFolderDictSamples(ImageFolder):
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
        super(ImageFolderDictSamples, self).__init__(root,
                                                     transform=transform,
                                                     target_transform=target_transform,
                                                     loader=loader,
                                                     is_valid_file=is_valid_file)

        # Convert samples from tuples to dicts
        sample_dicts = []
        for sample in self.samples:
            sample_dicts.append({"path": sample[0], "target": sample[1]})
        self.samples = sample_dicts

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
