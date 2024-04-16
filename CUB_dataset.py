"""
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from cub_config import BASE_DIR, N_ATTRIBUTES
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        pkl_file_paths,
        use_attr,
        no_img,
        uncertain_label,
        image_dir,
        n_class_attr,
        prune = False,
        transform=None,
        no_label=False,
    ):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            with open(file_path, "rb") as f:
                self.data.extend(pickle.load(f))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.no_label = no_label
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.prune = prune
        self.pruned_attr = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 310, 311]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]
        idx = img_path.split("/").index("CUB_200_2011")
        img_path = "/".join([self.image_dir] + img_path.split("/")[idx + 1 :])
        img = Image.open(img_path).convert("RGB")

        class_label = img_data["class_label"]
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data["uncertain_attribute_label"]
            else:
                attr_label = img_data["attribute_label"]

            attr_label = torch.tensor(attr_label).float()

            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    for index in range(N_ATTRIBUTES):
                        if img_data["uncertain_attribute_label"][index] != 0:
                            one_hot_attr_label[index][int(attr_label[index])] = 1
                        else:
                            one_hot_attr_label[index][2] = 1
                    one_hot_attr_label = torch.tensor(one_hot_attr_label).float()
                    return one_hot_attr_label, class_label
                else:
                    if self.prune:
                        attr_label = attr_label[self.pruned_attr]
                    return attr_label, class_label
            else:
                if self.no_label:
                    return img, attr_label
                else:
                    return img, class_label, attr_label
        else:
            return img, class_label


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]["attribute_label"][0]

    def __iter__(self):
        idx = (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )
        return idx

    def __len__(self):
        return self.num_samples


def load_data(
    pkl_paths,
    use_attr,
    no_img,
    batch_size,
    uncertain_label=False,
    n_class_attr=2,
    image_dir="images",
    resampling=False,
    resol=299,
):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    resized_resol = int(resol * 256 / 224)
    is_training = any(["train.pkl" in f for f in pkl_paths])
    if is_training:
        transform = transforms.Compose(
            [
                # transforms.Resize((resized_resol, resized_resol)),
                # transforms.RandomSizedCrop(resol),
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
                # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                # transforms.Resize((resized_resol, resized_resol)),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
                # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ]
        )

    dataset = CUBDataset(
        pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform
    )
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
        sampler = BatchSampler(
            ImbalancedDatasetSampler(dataset),
            batch_size=batch_size,
            drop_last=drop_last,
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
    return loader


def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    data = pickle.load(open(os.path.join(BASE_DIR, pkl_file), "rb"))
    n = len(data)
    n_attr = len(data[0]["attribute_label"])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d["attribute_label"]
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j] / n_ones[j] - 1)
    if not multiple_attr:  # e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio
