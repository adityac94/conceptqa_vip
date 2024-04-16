import os

import numpy as np
import torch
import torchvision

import CUB_dataset, custom_dataset


def _get_scale(square: bool) -> float:
    return 10 if square else 20


def db_to_ratio(db: float, square: bool = True) -> float:
    return 10 ** (db / _get_scale(square))


def ratio_to_db(ratio: float, square: bool = True) -> float:
    return _get_scale(square) * np.log10(ratio)


def get_data(dataset, transform):
    if dataset == "imagenet":
        train_ds = torchvision.datasets.DatasetFolder(
            os.path.join("./data/ImageNet_clip/train"),
            loader=lambda path: np.load(path, allow_pickle=True),
            is_valid_file=lambda path: path[-4:] == ".npy",
        )

        test_ds = torchvision.datasets.DatasetFolder(
            os.path.join("./data/ImageNet_clip/val"),
            loader=lambda path: np.load(path, allow_pickle=True),
            is_valid_file=lambda path: path[-4:] == ".npy",
        )

    if dataset == "places365":
        train_ds = custom_dataset.Places365_clip(
            root="data/Places365_clip",
            split="train-standard",
            small=True,
            download=False,
            transform=transform,
        )

        test_ds = custom_dataset.Places365_clip(
            root="data/Places365_clip",
            split="val",
            small=True,
            download=False,
            transform=transform,
        )

    if dataset == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    if dataset == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    elif dataset == "cub":
        use_attr = True
        no_img = False
        uncertain_label = False
        n_class_atr = 1
        data_dir = "data"
        image_dir = f"{data_dir}/CUB/CUB_200_2011"
        no_label = False
        prune = False

        train_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/trainclass_level_all_features.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=transform,
            no_label=no_label,
        )

        val_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/valclass_level_all_features.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=transform,
            no_label=no_label,
        )


        test_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/testclass_level_all_features.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=transform,
            no_label=no_label,
        )
        return train_ds, val_ds, test_ds

    return train_ds, test_ds


def get_concepts(filename):
    list_of_concepts = []

    f = open(filename, "r")
    for line in f.readlines():
        list_of_concepts.append(line.strip())

    return list_of_concepts

def compute_queries_needed(logits, threshold):
    """Compute the number of queries needed for each prediction.
    Parameters:
        logits (torch.Tensor): logits from querier
        threshold (float): stopping criterion, should be within (0, 1)

    """
    assert 0 < threshold and threshold < 1, 'threshold should be between 0 and 1'
    n_samples, n_queries, _ = logits.shape

    # turn logits into probability and find queried prob.
    prob = F.softmax(logits, dim=2)
    prob_max = prob.amax(dim=2)

    # `decay` to multipled such that argmax finds
    #  the first nonzero that is above threshold.
    threshold_indicator = (prob_max >= threshold).float().cuda()
    decay = torch.linspace(10, 1, n_queries).unsqueeze(0).cuda()
    semantic_entropy = (threshold_indicator * decay).argmax(1)

    # `threshold_indicator`==0 is to check which
    # samples did not stop querying, hence indicator vector
    # is all zeros, preventing bug that yields argmax as 0.
    semantic_entropy[threshold_indicator.sum(1) == 0] = n_queries
    semantic_entropy[threshold_indicator.sum(1) != 0] += 1

    return semantic_entropy


def compute_queries_needed_mi(logits, threshold, k=1):
    """Compute the number of queries needed for each prediction.

    Parameters:
        logits (torch.Tensor): logits from querier
        threshold (float): stopping criterion, should be within (0, 1)

    """
    n_samples, n_queries, _ = logits.shape

    # turn logits into probability and find queried prob.
    prob = F.softmax(logits, dim=2)

    entropy1 = -(prob[:, :-1] * np.log2(prob[:, :-1])).sum(dim=2)
    entropy2 = -(prob[:, 1:] * np.log2(prob[:, 1:])).sum(dim=2)

    difference = (np.absolute(entropy1 - entropy2))

    difference = torch.cat([difference, torch.zeros(difference.size(0), 1)], dim=1)

    # `decay` to multipled such that argmax finds
    #  the first nonzero that is above threshold.
    threshold_indicator = (difference <= threshold).float()

    signal = threshold_indicator.view(threshold_indicator.size(0), 1, -1)

    # convolution kernel of size 3, expecting 1 input channel and 1 output channel
    kernel = torch.ones(1, 1, k, requires_grad=False)
    # convoluting signal with kernel and applying padding
    output = F.conv1d(signal, kernel, stride=1, padding=k - 1, bias=None)[:, :, k - 1:].squeeze(1)

    threshold_indicator = (output == k).float()

    decay = torch.linspace(10, 1, n_queries).unsqueeze(0)
    semantic_entropy = (threshold_indicator * decay).argmax(1)

    # `threshold_indicator`==0 is to check which
    # samples did not stop querying, hence indicator vector
    # is all zeros, preventing bug that yields argmax as 0.
    semantic_entropy[threshold_indicator.sum(1) == 0] = n_queries
    semantic_entropy[threshold_indicator.sum(1) != 0] += 1

    return semantic_entropy

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is None:
               continue
            else:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def verbose_sequential(x, y, max_queries, actor, classifier):
    masked_image = torch.zeros(x.size()).cuda()
    mask = torch.zeros(x.size()).cuda()
    logits = []
    acc = []
    queries = []
    for i in range(max_queries + 1):
        query_vec = actor(masked_image, mask)
        label_logits = classifier(masked_image)
        mask[np.arange(x.size(0)), query_vec.argmax(dim=1)] = 1.0
        masked_image = masked_image + (query_vec * x)
        logits.append(label_logits)
        queries.append(query_vec)

        acc.append((label_logits.argmax(dim=1).float() == y.squeeze()).float().mean().cpu().item())

    return np.array(acc), torch.stack(logits).permute(1, 0, 2).cpu(), queries, masked_image


def clip_preprocess(tensors, size):
    transform = T.Compose([
        T.Resize(size=size, interpolation=BICUBIC),
        T.CenterCrop(size=size),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return transform(tensors)
