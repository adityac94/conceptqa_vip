
import numpy as np
import  sys
import torch
import utils
import torch.nn as nn
import pdb
import torch.optim as optim
import os
import wandb
import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import archs.cub_concept_net as cub_concept_net
from archs.network_cifar10 import DLA
from archs.vip_network import Network, ConceptNet2
import random
from archs.network_cifar100 import densenet201, densenet161, densenet169, densenet121
import train_concept_qa
import argparse

rs = 0 # random seed
Cosine_T_Max = 200
THRESHOLD = 0.85
EVAL_FREQUENCY = 1
EPS = 1.0
iterations_to_decrease_eps_over = 500

torch.set_num_threads(1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        return curr_lr

def get_answering_model(dataset_name, MAX_QUERIES):
    model = ConceptNet2()
    if dataset_name == "cifar10":
        model.load_state_dict(torch.load(f"saved_models/{dataset_name}/{dataset_name}_answers_clip_finetuned_classifier.pth"))

    elif dataset_name == "cub":
        model.load_state_dict(torch.load(f"saved_models/{dataset_name}/{dataset_name}_answers_clip_finetuned_classifier.pth"))

    elif dataset_name == "cifar100":
        model.load_state_dict(torch.load(f"saved_models/{dataset_name}/{dataset_name}_answers_clip_finetuned_classifier.pth"))

    elif dataset_name == "imagenet":
        model.load_state_dict(torch.load(f"saved_models/{dataset_name}/{dataset_name}_answers_clip_finetuned_depends_no_classifier_epoch_611.pth"))

    elif dataset_name == "places365":
        model.load_state_dict(
            torch.load(f"saved_models/{dataset_name}/{dataset_name}_answers_clip_finetuned_depends_no_classifier_epoch_1850.pth"))

    elif dataset_name == "cub_annotated":
        model.load_state_dict(torch.load(f"saved_models/{dataset_name}/{dataset_name}_answers_clip_finetuned_depends_no_classifier_epoch_1290.pth"))


    return model

def get_vip_networks(dataset_name, mode, MAX_QUERIES):
    if dataset_name == "cub_annotated":
        actor = Network(query_size=MAX_QUERIES, output_size=MAX_QUERIES, eps=EPS).cuda()
        classifier = Network(query_size=MAX_QUERIES, output_size=200, eps=None).cuda()

    elif dataset_name == "cub":
        actor = Network(query_size= MAX_QUERIES, output_size=MAX_QUERIES, eps=EPS).cuda()
        classifier = Network(query_size= MAX_QUERIES, output_size= 200, eps=None).cuda()

    elif dataset_name == "cifar10":
        actor = Network(query_size=MAX_QUERIES, output_size=MAX_QUERIES, eps=EPS).cuda()
        classifier = Network(query_size=MAX_QUERIES, output_size=10, eps=None).cuda()

    elif dataset_name == "cifar100":
        actor = Network(query_size=MAX_QUERIES, output_size=MAX_QUERIES, eps=EPS).cuda()
        classifier = Network(query_size=MAX_QUERIES, output_size=100, eps=None).cuda()

    elif dataset_name == "imagenet":
        actor = Network(query_size=MAX_QUERIES, output_size=MAX_QUERIES, eps=EPS).cuda()
        classifier = Network(query_size=MAX_QUERIES, output_size=1000, eps=None).cuda()

    elif dataset_name == "places365":
        actor = Network(query_size=MAX_QUERIES, output_size=MAX_QUERIES, eps=EPS).cuda()
        classifier = Network(query_size=MAX_QUERIES, output_size=365, eps=None).cuda()

    if mode == "biased":
        actor.load_state_dict(torch.load(f"saved_models/{dataset_name}/model_actor_{dataset_name}_vip_random.pth"))
        classifier.load_state_dict(torch.load(f"saved_models/{dataset_name}/model_classifier_{dataset_name}_vip_random.pth"))

    return actor, classifier

def get_max_queries(dataset_name):
    if dataset_name == "cub":
        return 208
    elif dataset_name == "cifar10":
        return 128

    elif dataset_name == "cifar100":
        return 824 

    elif dataset_name == "imagenet":
        return 4523 

    elif dataset_name == "places365":
        return 2207 

    elif dataset_name == "cub_annotated":
        return 312

def gen_histories(x, num_queries, actor):
    mask = torch.zeros(x.size()).cuda()
    final_mask = torch.zeros(x.size()).cuda()
    masked_image = torch.zeros(x.size()).cuda()
    final_masked_image = torch.zeros(x.size()).cuda()
    sorted_indices = num_queries.argsort()
    counter = 0

    with torch.no_grad():
        for i in range(MAX_QUERIES + 1):
            while (counter < x.size(0)):
                batch_index = sorted_indices[counter]
                if i == num_queries[batch_index]:
                    final_mask[batch_index] = mask[batch_index]
                    final_masked_image[batch_index] = masked_image[batch_index]
                    counter += 1
                else:
                    break
            if counter == x.size(0):
                break
            query_vec = actor(masked_image, mask)
            mask[np.arange(x.size(0)), query_vec.argmax(dim=1)] = 1.0
            masked_image = masked_image + (query_vec * x)
    return final_mask, final_masked_image

def query_entropy(query_vec, query_preds):
    query_preds = torch.nn.Sigmoid()(query_preds)

    entropy = -query_preds*torch.log(query_preds) - (1 - query_preds)*torch.log(1 - query_preds)

    return (entropy*query_vec).sum()/(entropy.size(0))

def get_input_features_for_concept_net(x, dataset_name, clip_model = None, arg_dictionary = None):
    global model_clip, dictionary
    if arg_dictionary is not None:
        dictionary = arg_dictionary

    if clip_model is not None:
        model_clip = clip_model

    if dataset_name in ["cub", "cifar10", "cifar100", "cub_annotated"]:
        image_features = model_clip.encode_image(x)
    elif dataset_name in ["imagenet", "places365"]:
        image_features = x.clone().cuda().half()
    image_features = image_features / torch.linalg.norm(image_features, axis=1).reshape(-1, 1)

    logit_scale = model_clip.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ dictionary

    max_val_per_image = logits_per_image.amax(dim=1).unsqueeze(1)
    min_val_per_image = logits_per_image.amin(dim=1).unsqueeze(1)
    logits_per_image = (logits_per_image - min_val_per_image) / (max_val_per_image - min_val_per_image)


    image_features = image_features.repeat(dictionary.size(1), 1, 1).permute(1, 0, 2)
    dictionary_extended = dictionary.T.repeat(image_features.size(0), 1, 1)

    input_features = torch.cat((image_features, dictionary_extended), dim=2)
    input_features = torch.flatten(input_features, 0, 1)

    return input_features.float(), logits_per_image.size()

def train(train_ds, val_ds, concept_net, dataset_name, discrete=False, mode="random"):
    n_epochs = NUM_EPOCHS

    criterion = nn.CrossEntropyLoss()

    print (len(val_ds), len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SZ, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SZ_TEST, num_workers=4)

    concept_net.eval()
    concept_net.cuda()

    actor, classifier = get_vip_networks(dataset_name, mode, MAX_QUERIES)

    actor = actor.cuda()
    classifier = classifier.cuda()

    if mode == "random":
        optimizer = optim.Adam(list(actor.parameters()) + list(classifier.parameters()), amsgrad=True, lr=1e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Cosine_T_Max)

    elif mode == "biased":
        optimizer1 = optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer2 = optim.Adam(actor.parameters(), lr=1e-4)

        scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[30, 60, 90, 120, 160, 200, 250], gamma=0.2)

        scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=Cosine_T_Max)

    eps_vals = np.linspace(EPS, 0.2, iterations_to_decrease_eps_over)
    for epoch_ind in range(n_epochs):
        # shuffle indices in batch
        actor.train()
        classifier.train()
        try:
            actor.change_eps(eps_vals[epoch_ind])
        except IndexError:
            pass
        for data in tqdm.tqdm(train_loader):
            if dataset_name in ["cub", "cub_annotated"]:
                x, y, c = data
            else:
                x, y = data

            if mode == "random":
                optimizer.zero_grad()
            elif mode == "biased":
                optimizer1.zero_grad()
                optimizer2.zero_grad()

            with torch.no_grad():
                input_features, logits_size = get_input_features_for_concept_net(x.cuda(), dataset_name)
                query_preds = concept_net(input_features).squeeze().view(logits_size)

            x = torch.where((query_preds > THRESHOLD_FOR_BINARIZATION), torch.ones(query_preds.size(), device=torch.device("cuda")), -torch.ones(query_preds.size(),  device=torch.device("cuda")))

            y = y.cuda()

            # different number of queries in a batch
            num_queries = torch.randint(low=0, high=MAX_QUERIES, size=(x.size(0),)) #substitute this line with max queries
            random_history = None

            if mode == "random":
                mask = torch.zeros(x.size()).cuda()

                for code_ind, num in enumerate(num_queries):
                    if num == 0:
                        continue
                    random_history = torch.multinomial(torch.ones(x.size(1)), num, replacement=False)
                    mask[code_ind, random_history.flatten()] = 1.0

                masked_answers = x * mask

            elif mode == "biased":
                mask, masked_answers = gen_histories(x, num_queries, actor)

            query_vec = actor(masked_answers, mask)

            updated_masked_answers = masked_answers + (query_vec * x)

            # another forward pass on architecture
            train_logits_cls = classifier(updated_masked_answers)

            if not discrete:
                loss = criterion(train_logits_cls.squeeze(), y.squeeze())
            else:
                loss = criterion(train_logits_cls, y.squeeze().long())

            entropy_regularizer = query_entropy(query_vec, query_preds)

            (loss + REG_WEIGHT*entropy_regularizer).backward()

            if mode == "random":
                wandb.log({'Train Cross Entropy': loss, 'selected_query_entropy': entropy_regularizer,
                           'train_lr': get_lr(optimizer), 'grad_norm_actor': utils.get_grad_norm(actor),
                                      'grad_norm_classifier': utils.get_grad_norm(classifier), 'eps': actor.eps})
                optimizer.step()

            elif mode == "biased":
                wandb.log({'Train Cross Entropy': loss, 'selected_query_entropy': entropy_regularizer, 'train_lr_1': get_lr(optimizer1),
                           'train_lr_2': get_lr(optimizer2), 'grad_norm_actor': utils.get_grad_norm(actor),
                           'grad_norm_classifier': utils.get_grad_norm(classifier), 'eps': actor.eps})

                optimizer1.step()
                optimizer2.step()

        if mode == "random":
            scheduler.step()

            torch.save(optimizer.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_optimizer.pth")
            torch.save(scheduler.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_scheduler.pth")

        elif mode == "biased":
            scheduler1.step()
            scheduler2.step()

            torch.save(optimizer1.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_optimizer1.pth")
            torch.save(scheduler1.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_scheduler1.pth")
            torch.save(optimizer2.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_optimizer2.pth")
            torch.save(scheduler2.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_scheduler2.pth")

        torch.save(actor.state_dict(), f"saved_models/{dataset_name}/model_actor_" + wandb.run.name + ".pth")
        torch.save(classifier.state_dict(), f"saved_models/{dataset_name}/model_classifier_" + wandb.run.name + ".pth")

        # evaluate test_loss every 10 epochs
        if epoch_ind % EVAL_FREQUENCY != 0:
            continue
        with torch.no_grad():
            epoch_loss, epoch_max_queries_accuracy, epoch_test_pred_ip, epoch_se, epoch_se_std = 0, 0, 0, 0, 0
            epoch_regularizer, epoch_acc = 0, 0
            counter = 0

            actor.eval()
            classifier.eval()

            all_outputs = []
            all_class_predictions = []

            for data in tqdm.tqdm(val_loader):

                if dataset_name in ["cub", "cub_annotated"]:
                    x, y, c = data
                else:
                    x, y = data

                with torch.no_grad():
                    input_features, logits_size = get_input_features_for_concept_net(x.cuda(), dataset_name)
                    query_preds = concept_net(input_features).squeeze().view(logits_size)

                x = torch.where((query_preds > THRESHOLD_FOR_BINARIZATION), torch.ones(query_preds.size()).cuda(),
                                -torch.ones(query_preds.size()).cuda())
                y = y.cuda()

                all_outputs += list(y.cpu().numpy())

                predicted_label = classifier(x).argmax(dim=1)

                all_class_predictions += list(predicted_label.cpu().numpy())

                # different number of queries in a batch
                num_queries = torch.randint(low=0, high=MAX_QUERIES, size=(x.size(0),))
                random_history = None
                mask = torch.zeros(x.size()).cuda()

                for code_ind, num in enumerate(num_queries):
                    if num == 0:
                        continue
                    random_history = torch.multinomial(torch.ones(x.size(1)), num, replacement=False)
                    mask[code_ind, random_history.flatten()] = 1.0

                masked_answers = x * mask
                query_vec = actor(masked_answers, mask)

                updated_masked_answers = masked_answers + (query_vec * x)

                test_logits_cls = classifier(updated_masked_answers)
                if not discrete:
                    loss = criterion(test_logits_cls.squeeze(), y.squeeze())
                    epoch_loss += torch.sqrt(loss)
                else:
                    loss = criterion(test_logits_cls, y.squeeze().long())
                    epoch_loss += loss * (x.size(0) / len(val_ds))
                    pred = test_logits_cls.argmax(dim=1).float()
                    epoch_acc += (pred == y.squeeze()).float().mean().item() * (x.size(0) / len(val_ds))

                # testing IP
                max_queries_accuracy, test_pred_ip, se, se_std = sequential(x, y, T_MAX, actor, classifier, len(val_ds),
                                                                            threshold=THRESHOLD)

                epoch_max_queries_accuracy += max_queries_accuracy
                epoch_test_pred_ip += test_pred_ip
                epoch_se += se
                epoch_se_std += se_std
                counter += 1
                # if counter == EPOCHS_TO_TEST:
                #     break

            if not discrete:
                wandb.log({'Test RMSE': epoch_loss})
            elif discrete:
                wandb.log({'Test Cross Entropy': epoch_loss,
                           'Test accuracy': epoch_acc})  # /(test_x.size(0)/BATCH_SZ_TEST)})

            epoch_max_queries_accuracy = epoch_max_queries_accuracy  # /(test_x.size(0)/BATCH_SZ_TEST)
            epoch_test_pred_ip = epoch_test_pred_ip  # / (test_x.size(0) / BATCH_SZ_TEST)
            epoch_se = epoch_se  # / (test_x.size(0) / BATCH_SZ_TEST)
            epoch_se_std = epoch_se_std  # / (test_x.size(0) / BATCH_SZ_TEST)

            if epoch_max_queries_accuracy > actor.current_max:
                print(epoch_max_queries_accuracy)
                actor.current_max = epoch_max_queries_accuracy
                torch.save(actor.state_dict(), f"saved_models/{dataset_name}/model_actor_" + wandb.run.name + "_best.pth")
                torch.save(classifier.state_dict(), f"saved_models/{dataset_name}/model_classifier_" + wandb.run.name + "_best.pth")

                if mode == "random":
                    torch.save(optimizer.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_optimizeri_best.pth")
                    torch.save(scheduler.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_scheduler_best.pth")

                elif mode == "biased":
                    torch.save(optimizer1.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_optimizer1_best.pth")
                    torch.save(scheduler1.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_scheduler1_best.pth")
                    torch.save(optimizer2.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_optimizer2_best.pth")
                    torch.save(scheduler2.state_dict(), f"saved_models/{dataset_name}/model_" + wandb.run.name + "_scheduler2_best.pth")

            all_outputs = np.array(all_outputs)
            all_class_predictions = np.array(all_class_predictions)

            wandb.log(
                {'All_queries_accuracy': (all_outputs == all_class_predictions).mean(), 'Max queries accuracy': epoch_max_queries_accuracy, "ip_accuracy": epoch_test_pred_ip, "se": epoch_se,
                 "se_std": epoch_se_std, "Epoch": epoch_ind})



def main(dataset_name, answering_model, mode):
    if dataset_name in ["cub", "cub_annotated"]:
        train_ds, val_ds, test_ds = utils.get_data(dataset_name, preprocess)
    else:
        train_ds, test_ds = utils.get_data(dataset_name, preprocess)

    if dataset_name in ["cub", "cub_annotated"]:
        train(train_ds, val_ds, answering_model, dataset_name, discrete = True, mode=mode)

    else:
        train(train_ds, test_ds, answering_model, dataset_name, discrete = True, mode=mode)

def sequential(x, y, max_queries, actor, classifier, dataset_size, threshold = 0.85):
    masked_image = torch.zeros(x.size()).cuda()
    mask = torch.zeros(x.size()).cuda()
    logits = []
    queries = []
    for i in range(max_queries):
        query_vec = actor(masked_image, mask)
        label_logits = classifier(masked_image)
        mask[np.arange(x.size(0)), query_vec.argmax(dim=1)] = 1.0
        masked_image = masked_image + (query_vec*x)
        logits.append(label_logits)
        queries.append(query_vec)

    max_queries_accuracy = (label_logits.argmax(dim=1).float() == y.squeeze()).float().mean().item() * (
                x.size(0) / dataset_size)

    logits = torch.stack(logits).permute(1, 0, 2)
    queries_needed = utils.compute_queries_needed(logits, threshold=threshold)

    test_pred_ip = logits[torch.arange(len(queries_needed)), queries_needed - 1].argmax(1)

    ip_accuracy = (test_pred_ip == y.squeeze()).float().mean().item() * (x.size(0) / dataset_size)

    se = queries_needed.float().mean().item() * (x.size(0) / dataset_size)
    se_std = queries_needed.float().std().item() * (x.size(0) / dataset_size)

    return max_queries_accuracy, ip_accuracy, se, se_std

if __name__ == "__main__":
    """ This is executed when run from the command line """

    import GPUtil

    rs = 0
    torch.manual_seed(rs)
    random.seed(rs)
    np.random.seed(rs)

    devices = GPUtil.getAvailable(limit=float("inf"), maxLoad=0.1, maxMemory=0.05)
    print(", ".join([str(d) for d in devices]))
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(d) for d in devices])
    
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
    parser.add_argument(
        "-mode",
        "--training_mode",
        type=str,
        required=True,
        choices=["random", "biased"],
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name

    mode = args.training_mode

    BATCH_SZ = args.batch_size
    BATCH_SZ_TEST = args.batch_size

    NUM_EPOCHS = args.num_epochs

    if dataset_name in ["cub", "cub_annotated", "imagenet", "places365"]:
        THRESHOLD_FOR_BINARIZATION = -0.4

    else:
        THRESHOLD_FOR_BINARIZATION = 0.0

    MAX_QUERIES = get_max_queries(dataset_name)

    answering_model = get_answering_model(dataset_name, MAX_QUERIES)

    T_MAX = 20#MAX_QUERIES + 1
    REG_WEIGHT = 0.0


    wandb.init(project="LLM_VIP", name=f"{dataset_name}_vip_{mode}", reinit=True)

    import clip

    model_clip, preprocess = clip.load("ViT-B/16", device=torch.device("cuda"))

    with torch.no_grad():
        concepts = utils.get_concepts("./concept_sets/" + dataset_name + ".txt")
        text = clip.tokenize(concepts).to(torch.device("cuda"))
        text_features = model_clip.encode_text(text)
        dictionary = text_features.T

        dictionary = dictionary / torch.linalg.norm(dictionary, axis=0)

    main(dataset_name, answering_model, mode)
