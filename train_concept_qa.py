import numpy as np
import torch
import argparse
import os
from utils import get_data, clip_preprocess
import tqdm
import random
from archs.network_cifar10 import DLA
from archs.network_cifar100 import densenet201, densenet161, densenet169, densenet121
import wandb
import pdb
import torch.optim as optim
from archs.vip_network import Network, ConceptNet2
from sklearn.metrics import precision_score, recall_score
import archs.cub_concept_net as cub_concept_net
import torchvision
from focal_loss.focal_loss import FocalLoss

torch.set_num_threads(1)

def get_model(dataset_name, num_classes):
    model = ConceptNet2() #Network(512+512, 1, batchnorm=True)
    return model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        return curr_lr

def get_optimizer(dataset_name, model):
    if dataset_name == "cifar10":
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    elif dataset_name == "cifar100":
        return optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

    elif dataset_name in ["cub", "cub_annotated"]:
        return optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    elif dataset_name == "imagenet":
        return optim.Adam(model.parameters(), amsgrad=True, lr=1e-4)

    elif dataset_name == "places365":
        return optim.Adam(model.parameters(), amsgrad=True, lr=1e-4)

def get_scheduler(dataset_name, optimizer):
    if dataset_name == "cifar100":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 260, 320, 360], gamma=0.2)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return scheduler


def get_gpt_answers(dataset_name):
    if dataset_name == "cifar10":
        return np.load("./gpt_answers/cifar10_answers_gpt4.npy")

    elif dataset_name == "cifar100":
        return np.load("./gpt_answers/cifar100_answers_gpt3.5_new_2.npy")

    elif dataset_name == "imagenet":
        return np.load("./gpt_answers/imagenet_answers_gpt3.5.npy")

    elif dataset_name == "places365":
        return np.load("./gpt_answers/places365_answers_gpt3.5.npy")

    elif dataset_name == "cub":
        return np.load("./gpt_answers/cub_answers_gpt4.npy")

    elif dataset_name == "cub_annotated":
        return np.load("./gpt_answers/cub_annotated_answers_gpt4.npy")


def train(dataset, bs, dataset_name, model, optimizer, scheduler, gpt_answers):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)
    criterion = torch.nn.BCEWithLogitsLoss(reduce=False)
    # criterion = FocalLoss(gamma=5.0, reduction="none")
    model.train()

    for batch_i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):

        optimizer.zero_grad()

        if dataset_name in ["cub", "cub_annotated"]:
            image, labels, _ = data
        else:
            image, labels = data

        query_answers = gpt_answers[labels]

        query_answers = np.where(query_answers == -1, np.zeros(query_answers.shape), query_answers)

        #changing all depends answers to a 1
        if dataset_name in ["cub", "cub_annotated", "cifar10", "cifar100"]:
            query_answers = np.where(query_answers == 2, np.ones(query_answers.shape), query_answers)
        else:
            query_answers = np.where(query_answers == 2, np.zeros(query_answers.shape), query_answers)

        query_answers = torch.tensor(query_answers).cuda()

        with torch.no_grad():
            if dataset_name in ["cub", "cub_annotated", "cifar10", "cifar100"]:
                image_features = model_clip.encode_image(image.cuda())
            elif dataset_name in ["imagenet", "places365"]:
                image_features = image.clone().cuda().half()

            image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)

            logit_scale = model_clip.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ dictionary

            max_val_per_image = logits_per_image.amax(dim=1).unsqueeze(1)
            min_val_per_image = logits_per_image.amin(dim=1).unsqueeze(1)
            logits_per_image = (logits_per_image - min_val_per_image)/(max_val_per_image - min_val_per_image)

        image_features = image_features.repeat(dictionary.size(1), 1, 1).permute(1, 0, 2)
        dictionary_extended = dictionary.T.repeat(image_features.size(0), 1, 1)

        input_features = torch.cat((image_features, dictionary_extended), dim=2)
        input_features = torch.flatten(input_features, 0, 1)

        output = model(input_features.float()).squeeze().view(logits_per_image.size())

        log_positive_pred = torch.log(torch.nn.Sigmoid()(output))
        log_negative_pred = torch.log(1 - torch.nn.Sigmoid()(output))

        loss = log_positive_pred*(query_answers*logits_per_image) + log_negative_pred*((1 - query_answers) + query_answers*(1 - logits_per_image))

        loss = -loss.sum()/torch.numel(loss)

        loss.backward()
        optimizer.step()

        wandb.log({'Train Cross Entropy': loss.item(), 'train_lr': get_lr(optimizer)})

def test(dataset, dataset_name, model, gpt_answers, mode="val"):
    bs = 128

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4)
    model.eval()

    accuracy = 0
    precision = 0
    recall = 0

    global CURRENT_BEST

    for batch_i, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if dataset_name in ["cub", "cub_annotated"]:
            image, labels, _ = data
        else:
            image, labels = data

        query_answers = gpt_answers[labels]

        query_answers = np.where(query_answers == -1, np.zeros(query_answers.shape), query_answers)

        query_answers = np.where(query_answers == 2, np.zeros(query_answers.shape), query_answers) #this is just so BCEloss doesnt complain these values would be masked anyway in the loss

        # this mask is useless currently, change the first np.ones to np.zeros to make it useful
        mask_depends = np.where(query_answers == 2, np.ones(query_answers.shape), np.ones(query_answers.shape))

        query_answers = torch.tensor(query_answers, device=torch.device("cuda"))

        with torch.no_grad():
            if dataset_name in ["cub", "cub_annotated", "cifar10", "cifar100"]:
                image_features = model_clip.encode_image(image.cuda())
            elif dataset_name in ["imagenet", "places365"]:
                image_features = image.clone().cuda().half()

            image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)

            logit_scale = model_clip.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ dictionary

            max_val_per_image =  logits_per_image.amax(dim=1).unsqueeze(1)
            min_val_per_image = logits_per_image.amin(dim=1).unsqueeze(1)
            logits_per_image = (logits_per_image - min_val_per_image)/(max_val_per_image - min_val_per_image)

            image_features = image_features.repeat(dictionary.size(1), 1, 1).permute(1, 0, 2)
            dictionary_extended = dictionary.T.repeat(image_features.size(0), 1, 1)

            input_features = torch.cat((image_features, dictionary_extended), dim=2)
            input_features = torch.flatten(input_features, 0, 1)

            output = model(input_features.float()).squeeze().view(logits_per_image.size())

            model_answers = torch.where(output > 0, torch.ones(output.size(), device=torch.device("cuda")), torch.zeros(output.size(), device=torch.device("cuda")))

            accuracy += ((model_answers == query_answers).float().cpu()*mask_depends).sum()/(mask_depends.sum())
            precision += precision_score(query_answers.cpu().numpy().flatten(), model_answers.cpu().numpy().flatten(), sample_weight = mask_depends.flatten())
            recall += recall_score(query_answers.cpu().numpy().flatten(), model_answers.cpu().numpy().flatten(), sample_weight = mask_depends.flatten())

    wandb.log({mode + "_" + 'Accuracy': accuracy/(batch_i + 1), mode + "_" +'Precision': precision/(batch_i + 1), mode + "_" +'Recall': recall/(batch_i + 1)})

    if accuracy/(batch_i + 1) >= CURRENT_BEST:
        CURRENT_BEST = accuracy/(batch_i + 1)
        torch.save(model.state_dict(), f"saved_models/{dataset_name}/{wandb.run.name}_classifier_best.pth")

def main(dataset_name, num_epochs, batch_size):
    if dataset_name in ["cub", "cub_annotated"]:
        train_ds, val_ds, test_ds = get_data(dataset_name, preprocess)
    else:
        train_ds, test_ds = get_data(dataset_name, preprocess)

    gpt_answers = get_gpt_answers(dataset_name)

    print (dataset_name, gpt_answers.shape)

    model = get_model(dataset_name, num_classes = gpt_answers.shape[1]).cuda()
    optimizer = get_optimizer(dataset_name, model)
    scheduler = get_scheduler(dataset_name, optimizer)

    for epoch in range(num_epochs):
        train(train_ds, batch_size, dataset_name, model, optimizer, scheduler, gpt_answers)
        scheduler.step()
        with torch.no_grad():
            if dataset_name == "cub":
                test(val_ds, dataset_name, model, gpt_answers, mode="val")
            else:
                test(test_ds, dataset_name, model, gpt_answers)
        wandb.log({'Epoch': epoch + 1})

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"saved_models/{dataset_name}/{wandb.run.name}_classifier.pth")

def get_concepts(filename):
    list_of_concepts = []

    f = open(filename, "r")
    for line in f.readlines():
        list_of_concepts.append(line.strip())

    return list_of_concepts

if __name__ == '__main__':
    import GPUtil

    rs = 0
    torch.manual_seed(rs)
    random.seed(rs)
    np.random.seed(rs)

    devices = GPUtil.getAvailable(limit=float("inf"), maxLoad=0.1, maxMemory=0.05)
    print(", ".join([str(d) for d in devices]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(d) for d in devices])

    import clip

    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", "--num_epochs", type=int, default=4000)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument(
        "-dataset",
        "--dataset_name",
        type=str,
        required=True,
        choices=["imagenet", "places365", "cub", "cifar10", "cifar100"],
    )
    args = parser.parse_args()
    
    dataset_name = args.dataset_name

    model_clip, preprocess = clip.load("ViT-B/16", device=torch.device("cuda"))

    CURRENT_BEST = 0

    with torch.no_grad():
        concepts = get_concepts("./concept_sets/" + dataset_name + ".txt")
        text = clip.tokenize(concepts).to(torch.device("cuda"))
        text_features = model_clip.encode_text(text)
        dictionary = text_features.T

        dictionary = dictionary / torch.linalg.norm(dictionary, axis=0)

    wandb.init(project="LLM_VIP", name=f"{dataset_name}_conceptqa", reinit=True)
    main(dataset_name, args.num_epochs, args.batch_size)
