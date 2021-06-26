import os, argparse, json
import copy
from typing import List, Dict, Tuple, Set
import random

"""
Schedule parallel tasks to processors using a genetic algorithm

The tasks are not divisible and can be executed in any order, but you can put
constraints which specify that a task can only be run on a certain processor.

Input data is in the form of a JSON file. You should specify the number of
available processors, task names, and their execution times as a dictionary
under the "data" key like this:

    "data": {
        "num_processors": 3,
        "execution_times": {"t1": 2, "t2": 3, "t3": 3, "t4": 4,
                            "t5": 5, "t6": 4, "t7": 4, "t8": 4, "t9": 1},
        "constraints": {"t1": 0, "t9": 2}
    }

See the example .json files in this repo for more parameters.

"""

class ConstraintViolationException(Exception):
    pass

class Solution:
    completion_times = None
    num_processors = None
    constraints = None
    def __init__(self, sched: List[Tuple[str, int]], ignore_constraints=False):
        if not all([e is not None for e in [Solution.completion_times, Solution.num_processors, Solution.constraints]]):
            raise ValueError("Please set the completion times, the number of processors and the constraints first")
        # sprawdzenie czy ograniczenia są spełnione:
        if ignore_constraints is False:
            for task in sched:
                if task[0] in Solution.constraints.keys():
                    if Solution.constraints[task[0]] != task[1]:
                        raise ConstraintViolationException(f"Task {task[0]} cannot be assigned to processor {task[1]}.")
        self._schedule = sched

    @property
    def num_tasks(self):
        return len(self._schedule)

    @property
    def schedule(self):
        parallel_schedule = [list() for i in range(Solution.num_processors)]
        for task in self._schedule:
            parallel_schedule[task[1]].append((task[0], Solution.completion_times[task[0]]))
        return parallel_schedule

    @property
    def completion_time(self):
        execution_times = [sum([t[1] for t in processor]) for processor in self.schedule] # completion time for all tasks on each processor
        earliest_completion = max(execution_times) # earliest schedule completion time assuming parallel execution
        return execution_times, earliest_completion

    def __hash__(self):
        return str(self._schedule).__hash__()

    def __lt__(self, other):
        return self.completion_time[1] < other.completion_time[1]

    def __eq__(self, other):
        return self.completion_time[1] == other.completion_time[1]

    def __repr__(self):
        ret_str = f"CT={self.completion_time[1]}" 
        for processor, schedule in enumerate(self.schedule):
            ret_str += (f" <P{processor}: {schedule}>")
        return ret_str


def crossover_map(solution1: Solution, solution2: Solution, ignore_constraints=False):
    """ Swap processor assignment between `solution1` and `solution2` after a random crossover point. """
    crossover_point = random.randint(0, solution1.num_tasks)

    print(f"Crossover (map) after index {crossover_point}")
    processors1 = [t[1] for t in solution1._schedule[crossover_point:]]
    processors2 = [t[1] for t in solution2._schedule[crossover_point:]]
    offspring1 = list(zip([t[0] for t in solution1._schedule[crossover_point:]], processors2))
    offspring2 = list(zip([t[0] for t in solution2._schedule[crossover_point:]], processors1))

    offspring1 = solution1._schedule[0:crossover_point] + offspring1
    offspring2 = solution2._schedule[0:crossover_point] + offspring2

    try:
        new_solution1 = Solution(offspring1, ignore_constraints)
    except ConstraintViolationException as e:
        print(str(e))
        new_solution1 = None
    try:
        new_solution2 = Solution(offspring2, ignore_constraints)
    except ConstraintViolationException as e:
        print(str(e))
        new_solution2 = None
    return new_solution1, new_solution2


def crossover_order(solution1: Solution, solution2: Solution, ignore_constraints=False):
    """ Mix the task execution order of `solution1` and `solution2` """
    crossover_point = random.randint(0, solution1.num_tasks)

    print(f"Crossover (order) after index {crossover_point}")
    tasks1 = [t[0] for t in solution1._schedule[0:crossover_point]]
    processors1 = [t[1] for t in solution1._schedule]
    tasks2 = [t[0] for t in solution2._schedule]

    reordered = tasks1 + [t2 for t2 in tasks2 if t2 not in tasks1]
    try:
        reordered_solution = Solution(list(zip(reordered, processors1)), ignore_constraints=ignore_constraints)
    except ConstraintViolationException as e:
        print(str(e))
        reordered_solution = None
    return reordered_solution


def mutate(solution: Solution, mu_m: float, ignore_constraints=False):
    """ Randomly reassign tasks between processors """
    mutated = False
    for i, task in enumerate(solution._schedule):
        rnd = random.random()
        if rnd < mu_m:
            reassigned_processor = random.randint(0, Solution.num_processors-1)
            if ignore_constraints is False:
                if task[0] in Solution.constraints.keys():
                    if Solution.constraints[task[0]] != reassigned_processor:
                        print(f"Task {task[0]} cannot be reassigned to processor {reassigned_processor}.")
                        continue
            mutated = True
            new_assignment = (task[0], reassigned_processor)
            solution._schedule[i] = new_assignment
    return mutated

def initialize(*, num_processors: int, execution_times: Dict[str, int], constraints: Dict[str, int],
               population_count: int=20):
    """ Generate random `population_count` assignments s.t. constraints """
    Solution.num_processors = num_processors
    Solution.completion_times = execution_times
    Solution.constraints = constraints
    random_population = set()
    while len(random_population) < population_count:
        order = copy.deepcopy(list(execution_times.keys()))
        random.shuffle(order)
        assignment = random.choices(range(num_processors), k=len(order))
        try:
            solution = Solution(list(zip(order, assignment)))
        except ConstraintViolationException as e:
            print(str(e))
            continue
        random_population.add(solution)
    return random_population

def binary_tournament_selection(*, population: Set[Solution], k: int=2):
    """ Select `k` chromosomes/solutions using the binary tournament method

        We randomly sample two chromosomes from the population. Whichever has
        a better fit (i.e. a shorter total completion time) is selected for
        crossover and mutation.
    """
    candidates = []
    for i in range(k):
        two_chromosomes = random.sample(population, k=k)
        selected = two_chromosomes[0] if two_chromosomes[0] > two_chromosomes[1] else two_chromosomes[1]
        candidates.append(selected)
    return candidates

def main(cfg, run_times):

    iterresults = []
    for k in range(run_times):
        args = json.load(open(argz['config'], 'r'))
        print(args)
        population = initialize(num_processors=args['data']['num_processors'],
                                execution_times=args['data']['execution_times'],
                                constraints=args['data']['constraints'],
                                population_count=args['initial_population'])
        best = sorted(population)[0].completion_time[1]
        best_in_epoch = 0
        initial_best = best
        print(f"Initial best solution: {initial_best}")
        epoch_counter = 1
        epochs_without_improvement = 0
        while epochs_without_improvement < args['stop_after'] and epoch_counter < args['hard_stop']:
            print(f"================ EPOCH {epoch_counter} =================")
            found_better = False
            for step in range(args['steps_per_epoch']):
                selected = binary_tournament_selection(population=population, k=2)
                RN = random.random()
                if RN < args['mu_c']:
                    pass
                else:
                    RN2 = random.random()
                    if RN2 < 0.5:
                        offspring1, offspring2 = crossover_map(selected[0], selected[1])
                    else:
                        offspring1 = crossover_order(selected[0], selected[1])
                        offspring2 = None
                    for offspring in [offspring1, offspring2]:
                        if offspring is not None:
                            mutated = mutate(offspring, args['mu_m'])
                            if mutated: print(f"******* MUTATION *********")
                            population.add(offspring)
                            ct = offspring.completion_time[1]
                            if ct < best:
                                print(f">>>>>>>>>> FOUND A BETTER SCHEDULE WITH COMPLETION TIME OF {ct} <<<<<<<<<<<<<")
                                found_better = True
                                best = ct
                                best_in_epoch = epoch_counter
            if found_better is True:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            population = set(sorted(population)[:args['max_population']])
            epoch_counter += 1
        ret = sorted(population)[0]
        iterresults.append((epoch_counter, ret.completion_time[1], initial_best, best_in_epoch, ret))

    for k in iterresults:
        epoch_counter = k[0]
        completion_time = k[1]
        initial_best = k[2]
        best_in_epoch = k[3]
        ret = k[4]
        print(f"===================== SUMMARY =========================")
        print(f"After {epoch_counter} epochs found the best schedule with a value of {completion_time} (initial solution: {initial_best}) in epoch {best_in_epoch}.")
        print(f"The schedule is as follows:")
        print(ret)
        print(f"=======================================================")

    for k in iterresults:
        epoch_counter = k[0]
        completion_time = k[1]
        initial_best = k[2]
        best_in_epoch = k[3]
        ret = k[4]
        print(f"{best_in_epoch};{initial_best}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schedule parallel tasks on processors using a simple "
                                                 "genetic algorithm")
    parser.add_argument('--config', required=True, help='Path to a .json file with problem description')
    parser.add_argument('--run-times', type=int, default=1, help='Number of times the algorithm should run')
    argz = vars(parser.parse_args())
    main(argz['config'], argz['run_times'])
