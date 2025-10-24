import copy
import random
from tqdm import tqdm
import numpy as np
import itertools
import torch

from classification_evaluation import validate_one_epoch_accuracy

__all__ = ["EvolutionFinder"]


class ArchManager:
    def __init__(self):
        self.num_choice_layer_tuple = (4, 4, 4, 4)
        self.num_choice_layer = len(self.num_choice_layer_tuple)

    def random_sample(self):
        '''
        The generates a architecture with a set of masks
        '''

        choice_dict = {
            'choice_1': random.randrange(self.num_choice_layer_tuple[0]),
            'choice_2': random.randrange(self.num_choice_layer_tuple[1]),
            'choice_3': random.randrange(self.num_choice_layer_tuple[2]),
            'choice_4': random.randrange(self.num_choice_layer_tuple[3])

        }

        return choice_dict

    def random_resample(self, choice_dict, i):
        assert i >= 1 and i <= self.num_choice_layer
        choice_dict[f'choice_{i}'] = random.randrange(self.num_choice_layer_tuple[i - 1])


class EvolutionFinder:
    def __init__(
            self,
            accuracy_predictor,
            model,
            fake_loader,
            valid_loader,
            loss,
            metrics,
            device,
            args,
            **kwargs
    ):

        self.accuracy_predictor = accuracy_predictor
        self.arch_manager = ArchManager()
        self.num_choice_layer = self.arch_manager.num_choice_layer

        self.mutate_prob = kwargs.get("mutate_prob")#, 0.1)
        self.population_size = kwargs.get("population_size")
        self.max_time_budget = kwargs.get("max_time_budget")#, 500)
        self.parent_ratio = kwargs.get("parent_ratio")#, 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio")#, 0.5)

        self.model = model
        self.fake_loader = fake_loader
        self.valid_loader = valid_loader
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.args = args

    def random_sample(self):
        # while True:
        sample = self.arch_manager.random_sample()
        return sample

    def mutate_sample(self, sample):
        # while True:
        new_sample = copy.deepcopy(sample)

        for i in range(self.num_choice_layer):
            if random.random() < self.mutate_prob:
                self.arch_manager.random_resample(new_sample, i + 1)

        return new_sample

    def crossover_sample(self, sample1, sample2):
        # while True:
        new_sample = copy.deepcopy(sample1)

        for layer_choice in new_sample.keys():
            new_sample[layer_choice] = random.choice(
                [sample1[layer_choice], sample2[layer_choice]]
            )

        return new_sample

    def run_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size

        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []

        best_info = None
        best_info_list = [-100]

        if verbose:
            print("Generate random population...")
        for _ in tqdm(range(population_size)):
            sample = self.random_sample()
            child_pool.append(sample)
        print('child_pool finished')

        accs = []
        for i in tqdm(range(population_size)):
            #
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            np.random.seed(42)
            random.seed(42)
            torch.backends.cudnn.deterministic = True

            results_pred = self.accuracy_predictor(model=self.model, choice_dict=child_pool[i],
                                                   loader=self.valid_loader,
                                                   args=self.args)
            acc_pred = 100 - results_pred[0]
            ece_pred = results_pred[1]
            _, _, aPE_pred = self.accuracy_predictor(model=self.model, choice_dict=child_pool[i],
                                                     loader=self.fake_loader,
                                                     args=self.args)

            accs.append(#acc_pred #- self.args.weight_of_ece * ece_pred
                        acc_pred
                        - self.args.weight_of_ece * ece_pred
                        + self.args.weight_of_aPE * aPE_pred
                        )
            #

        print('calculate accuracy finished')
        # accs = self.accuracy_predictor(model=self.model, choice_dict=child_pool[0], valid_loader=self.valid_loader, loss=self.loss, metrics=self.metrics, device=self.device)
        # validate_one_epoch_accuracy(model=model, valid_loader=valid_loader, loss=criterion, metrics=accuracy,device=device)

        for i in tqdm(range(population_size)):
            population.append((accs[i], child_pool[i]))
            # population.append((accs[i].item(), child_pool[i]))
        print('population finished')

        if verbose:
            print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(range(max_time_budget)):
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]
            if verbose:
                print("Iter: {} Acc: {}".format(iter - 1, parents[0][0]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
                best_info_list.append(parents[0])
            else:
                best_valids.append(best_valids[-1])
                best_info_list.append(best_info_list[-1])

            population = parents
            child_pool = []

            for i in range(mutation_numbers):
                par_sample = population[np.random.randint(parents_size)][1]
                # Mutate
                new_sample = self.mutate_sample(par_sample)
                child_pool.append(new_sample)

            for i in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                # Crossover
                new_sample = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)

            accs = []
            for i in range(population_size):
                #
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                np.random.seed(42)
                random.seed(42)
                torch.backends.cudnn.deterministic = True
                #

                results_pred = self.accuracy_predictor(model=self.model, choice_dict=child_pool[i],
                                                       loader=self.valid_loader,
                                                       args=self.args)
                acc_pred = 100 - results_pred[0]
                ece_pred = results_pred[1]
                _, _, aPE_pred = self.accuracy_predictor(model=self.model, choice_dict=child_pool[i],
                                                         loader=self.fake_loader,
                                                         args=self.args)

                accs.append(#acc_pred #- self.args.weight_of_ece * ece_pred
                            acc_pred
                            - self.args.weight_of_ece * ece_pred
                            + self.args.weight_of_aPE * aPE_pred
                            )

            for i in range(population_size):
                population.append((accs[i], child_pool[i]))
                # population.append((accs[i].item(), child_pool[i]))

            # with open(f'data_{iter}.txt', 'w') as file:
            #     for item in population:
            #         file.write(str(item)+'\n')

        return best_valids, best_info, best_info_list
