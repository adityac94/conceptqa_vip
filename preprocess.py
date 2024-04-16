import argparse
import os

import clip
import CUB_dataset
import numpy as np
import torch
import torchvision
import tqdm

torch.set_num_threads(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)


def get_data(transform, dataset):
    if dataset == "places365":
        train_ds = torchvision.datasets.Places365(
            root="./data/Places365",
            split="train-standard",
            small=True,
            download=True,
            transform=transform,
        )

        test_ds = torchvision.datasets.Places365(
            root="./data/Places365",
            split="val",
            small=True,
            download=True,
            transform=transform,
        )

    if dataset == "imagenet":
        data_dir = "./data/ImageNet"
        train_ds = torchvision.datasets.ImageFolder(f"{data_dir}/train/", transform=preprocess)
        test_ds = torchvision.datasets.ImageFolder(f"{data_dir}/val/", transform=preprocess)


def clip_embedding(dataset, split="train", dataset_name=None):
    bs = 512

    with torch.no_grad():
        iteration = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4)

        print(dataset_name)

        for batch_i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            start_i, end_i = batch_i * bs, (batch_i + 1) * bs

            if dataset_name == "cub":
                image, _, _ = data
            else:
                image, _ = data
            image = image.to(device)

            image_features = model.encode_image(image).float()
            image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)

            image_features = image_features.cpu().numpy()
            iteration += 1

            # save path
            if dataset_name == "places365":
                for i, tup in enumerate(dataset.imgs[start_i:end_i]):
                    img_path = tup[0]

                    if split == "train":
                        alphabet, class_name, img_name = img_path.split("/")[-3:]
                        filename = img_name[:-4]
                        save_dir = f"./data/Places365_clip/train-standard/{alphabet}/{class_name}"
                    elif split == "val":
                        filename = img_path.split("/")[-1][:-4]
                        save_dir = "./data/Places365_clip/val_256/"
                    else:
                        raise NameError(f"split not found: {split}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{filename}.npy")
                    np.save(save_path, image_features[i])

            elif dataset_name == "imagenet":
                for i, tup in enumerate(dataset.imgs[start_i:end_i]):
                    img_path = tup[0]
                    img_dir = img_path.split("/")[-2]
                    filename = img_path.split("/")[-1][:-5]
                    save_dir = f"./data/ImageNet_clip/{split}/{img_dir}/"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{filename}.npy")
                    np.save(save_path, image_features[i])


def main(dataset):
    train_ds, test_ds = get_data(preprocess, dataset)
    clip_embedding(train_ds, "train", dataset_name=dataset)
    clip_embedding(test_ds, "val", dataset_name=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset_name", type=str, choices=["imagenet", "places365"])
    args = parser.parse_args()

    main(args.dataset_name)
