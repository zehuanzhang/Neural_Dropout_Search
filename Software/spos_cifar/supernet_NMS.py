import os

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys

import matplotlib.pyplot as plt

sys.path.append("./models")

import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn

from nni.nas.pytorch.callbacks import LRSchedulerCallback
from nni.nas.pytorch.callbacks import ModelCheckpoint
from nni.nas.pytorch.spos import SPOSSupernetTrainingMutator, SPOSSupernetTrainer

from utils import CrossEntropyLabelSmooth, accuracy

from models import ModelFactory
from software.losses import ClassificationLoss
from software.data import *

from classification_evaluation import classification_evaluation
from classification_evaluation_loader import classification_evaluation_loader, _evaluate_with_loader
from supernet_NMS_EvolutionFinder import EvolutionFinder

import json
import itertools
from tqdm import tqdm

import pickle

logger = logging.getLogger("nni.spos.supernet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPOS Supernet Training")
    parser.add_argument("--spos-preprocessing", action="store_true", default=False,
                        help="When true, image values will range from 0 to 255 and use BGR "
                             "(as in original repo).")
    parser.add_argument("--workers", type=int, default=1)  # 4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smooth", type=float, default=0.1)
    parser.add_argument("--log-frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    #
    parser.add_argument('--model', type=str, default='resnet_all',#lenet_all
                        help='the model that we want to train')
    parser.add_argument('--p', type=float,
                        default=0.25, help='dropout probability')
    parser.add_argument('--input_size', nargs='+',
                        # default=[1, 1, 28, 28], help='input size')
                        default=[1, 3, 32, 32], help='input size')
    parser.add_argument('--output_size', type=int,
                        default=10, help='output size')
    parser.add_argument('--q', action='store_true',
                        help='whether to do post training quantisation')
    parser.add_argument('--smoothing', type=float,
                        default=0.0, help='smoothing factor')
    parser.add_argument('--learning_rate', type=float,
                        default=0.005, help='init learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.00001, help='weight decay')
    parser.add_argument('--epochs', type=int, default=400,  # 100,
                        help='num of training epochs')

    parser.add_argument('--data', type=str, default='../data/',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar',#mnist
                        help='dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')#256

    parser.add_argument('--dataset_size', type=float,
                        default=1.0, help='portion of the whole training data')
    parser.add_argument('--valid_portion', type=float,
                        default=0.1, help='portion of training data')

    parser.add_argument("--model-dir", type=str, default="./checkpoints")
    parser.add_argument("--epoch", type=str, default="399")
    parser.add_argument("--samples", type=str, default=8)
    parser.add_argument("--debug", type=str, default=False)
    parser.add_argument("--save", type=str, default="results")

    parser.add_argument("--weight_of_ece", type=float, default=1)
    parser.add_argument("--weight_of_aPE", type=float, default=1)
    #

    args = parser.parse_args()
    print(torch.cuda.is_available(), torch.__version__)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    ####################################################################################################################
    model_temp = ModelFactory.get_model
    model = model_temp(args.model, args.input_size,
                       args.output_size, args)

    # print('args.load_checkpoint:', args.load_checkpoint)
    # if args.load_checkpoint:
    #     if not args.spos_preprocessing:
    #         logger.warning("You might want to use SPOS preprocessing if you are loading their checkpoints.")
    #     model.load_state_dict(load_and_parse_state_dict())
    model.cuda()
    # if torch.cuda.device_count() > 1:  # exclude last gpu, saving for data preprocessing on gpu
    #     model = nn.DataParallel(model, device_ids=list(range(0, torch.cuda.device_count() - 1)))

    ####################################################################################################################
    mutator = SPOSSupernetTrainingMutator(model)  # no flops limit

    criterion = ClassificationLoss(args)
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)
    train_loader, valid_loader = get_train_loaders(args)
    test_loader = get_test_loader(args)
    args.dataset = 'random_' + args.dataset
    fake_loader = get_test_loader(args)

    trainer = SPOSSupernetTrainer(model, criterion, accuracy, optimizer,
                                  args.epochs, train_loader, valid_loader,
                                  mutator=mutator, batch_size=args.batch_size,
                                  log_frequency=args.log_frequency, workers=args.workers,
                                  device='cuda:0',
                                  callbacks=[LRSchedulerCallback(scheduler),
                                             ModelCheckpoint("./checkpoints")])

    ####################################################################################################################
    train = False
    if train:
        trainer.train()

    ####################################################################################################################
    evolution = True
    if evolution:
        model_path = f"{args.model_dir}/epoch_{args.epoch}.pth.tar"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        P = 50  # The size of population in each generation
        N = 10 # How many generations of population to be searched
        r = 0.5  # The ratio of networks that are used as parents for next generation
        params = {
            'mutate_prob': 0.5,  # The probability of mutation in evolutionary search
            'mutation_ratio': 0.5,
            'population_size': P,
            'max_time_budget': N,
            'parent_ratio': r,
        }

        evolution_finder = EvolutionFinder(accuracy_predictor=_evaluate_with_loader, model=model,fake_loader=fake_loader,
                                           valid_loader=valid_loader, loss=criterion,
                                            metrics=accuracy, device=device, args=args, **params)
        evolution_results = evolution_finder.run_evolution_search() #(verbose=True)
        # print(evolution_results)

        result_tuple = evolution_results[1]#(accuracy, choice_dict)
        with open(f'./{args.save}/result_tuple.pickle', 'wb') as file:
            pickle.dump(result_tuple, file)

    ####################################################################################################################
    load = False

    with open(f'./{args.save}/result_tuple.pickle', 'rb') as file:
        acc, choice_dict = pickle.load(file)

    if load:
        model_path = f"{args.model_dir}/epoch_{args.epoch}.pth.tar"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        # specify seeds
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        # Search Result
        # print('Search Result:')
        error, ece, _ = classification_evaluation_loader(choice_dict, model, args,loader=valid_loader)  # test_loader valid_loader
        _, _, entropy = classification_evaluation_loader(choice_dict, model, args, loader=fake_loader)

        search_result = dict()
        search_result['choice_dict'] = choice_dict
        search_result['accuracy'] = 100 - error
        search_result['ece'] = ece
        search_result['entropy'] = entropy
        print('search_result:', search_result['accuracy'], search_result['ece'], search_result['choice_dict'])

        with open(f'./{args.save}/search_result_val.json', 'w') as f:
            json.dump(search_result, f)

        # All Results
        # print('All Results')
        num_choice_layer_tuple = (4, 4, 4, 4)
        choice_set_list = [
            {'choice_1': i, 'choice_2': j, 'choice_3': k, 'choice_4': l}
            for i, j, k, l in itertools.product(
                range(num_choice_layer_tuple[0]),
                range(num_choice_layer_tuple[1]),
                range(num_choice_layer_tuple[2]),
                range(num_choice_layer_tuple[3])
            )
        ]

        all_results_list = []
        for i in tqdm(range(len(choice_set_list))):
            choice_dict = choice_set_list[i]

            #specify seeds
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True

            error, ece, _ = classification_evaluation_loader(choice_dict, model, args, loader=valid_loader)  # test_loader valid_loader
            _, _, entropy = classification_evaluation_loader(choice_dict, model, args, loader=fake_loader)

            results = dict()
            results['choice_dict'] = choice_dict
            results['accuracy'] = 100 - error
            results['ece'] = ece
            results['entropy'] = entropy
            all_results_list.append(results)
            # print('all_result:', results['accuracy'], results['choice_dict'])

        with open(f'./{args.save}/all_results_list_val.json', 'w') as f:
            json.dump(all_results_list, f)
