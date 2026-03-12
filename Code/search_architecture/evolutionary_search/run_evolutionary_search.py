import os
import torch
import json
import random
import time
from tqdm import tqdm  # 用于显示进度条，pip install tqdm

# 复用您项目中的核心模块
from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel
from Code.search_architecture.lstm_search.Evaluator import LinkPredictionEvaluator
from Code.search_architecture.lstm_search.run_nas import load_and_split_data


def create_random_individual(MIN_LAYERS,MAX_LAYERS,CONV_CHOICES,NORM_CHOICES,ACT_CHOICES):
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    layer_configs = []
    for _ in range(num_layers):
        config = {
            'conv': random.choice(CONV_CHOICES),
            'norm': random.choice(NORM_CHOICES),
            'act': random.choice(ACT_CHOICES),
        }
        layer_configs.append(config)
    return {'arch': layer_configs, 'fitness': 0.0}

def evaluate_population(CHILD_TRAIN_EPOCHS,population, train_data, val_data, device):
    print("Starting population fitness evaluation...")
    for individual in tqdm(population, desc="Evaluating progress"):
        if individual['fitness'] > 0:
            continue
        try:
            model = FlexibleGNNModel(individual['arch'], train_data.metadata()).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.BCEWithLogitsLoss()
            evaluator = LinkPredictionEvaluator(model, criterion, optimizer, device)

            for _ in range(CHILD_TRAIN_EPOCHS):
                evaluator.train_one_epoch(train_data)

            val_metrics = evaluator.evaluate(val_data)
            reward = 0.5 * val_metrics['AUC'] + 0.5 * val_metrics['AUPR']
            individual['fitness'] = reward
        except Exception as e:
            print(f"Error occurred while evaluating architecture {individual['arch']}: {e}")
            individual['fitness'] = -1.0


def tournament_selection(TOURNAMENT_SIZE,population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: x['fitness'], reverse=True)
    return tournament[0]


def crossover(CROSSOVER_RATE,parent1, parent2):
    child1_arch, child2_arch = parent1['arch'][:], parent2['arch'][:]

    if random.random() < CROSSOVER_RATE:
        min_len = min(len(parent1['arch']), len(parent2['arch']))
        if min_len > 1:
            cut_point = random.randint(1, min_len - 1)
            child1_arch = parent1['arch'][:cut_point] + parent2['arch'][cut_point:]
            child2_arch = parent2['arch'][:cut_point] + parent1['arch'][cut_point:]

    return {'arch': child1_arch, 'fitness': 0.0}, {'arch': child2_arch, 'fitness': 0.0}


def mutate(individual,MUTATION_RATE,CONV_CHOICES,NORM_CHOICES,ACT_CHOICES):
    mutated_arch = individual['arch'][:]
    for i in range(len(mutated_arch)):
        if random.random() < MUTATION_RATE:
            mutated_arch[i]['conv'] = random.choice(CONV_CHOICES)
        if random.random() < MUTATION_RATE:
            mutated_arch[i]['norm'] = random.choice(NORM_CHOICES)
        if random.random() < MUTATION_RATE:
            mutated_arch[i]['act'] = random.choice(ACT_CHOICES)
    return {'arch': mutated_arch, 'fitness': 0.0}


def save_top_k_evolutionary(db_name, population, top_k):
    try:
        all_results_formatted = []
        population.sort(key=lambda x: x['fitness'], reverse=True)
        for i, individual in enumerate(population[:top_k]):
            formatted_res = {
                'rank': i + 1,
                'combined_reward': individual['fitness'],
                'architecture': individual['arch']
            }
            all_results_formatted.append(formatted_res)

        script_directory = 'Code/search_architecture/'
        results_dir = os.path.join(script_directory, f'results/{db_name.lower()}')
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, f"{db_name}_evolutionary.json")

        with open(filepath, 'w') as f:
            json.dump(all_results_formatted, f, indent=4)
        print(f"\nSuccess! Saved the top {len(all_results_formatted)} best architectures to {filepath}")

        return all_results_formatted
    except Exception as e:
        print(f"\n[ERROR] An exception occurred while saving the file: {e}")


def main():
    CONV_CHOICES = ['sage', 'graphconv', 'transformerconv', 'gatconv', 'gatv2conv']
    NORM_CHOICES = ['graphnorm', 'instancenorm', 'layernorm', 'batchnorm', 'identity']
    ACT_CHOICES = ['relu', 'tanh', 'leakyrelu', 'identity']

    POPULATION_SIZE = 20
    NUM_GENERATIONS = 10
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8
    ELITISM_COUNT = 2
    TOURNAMENT_SIZE = 3
    MIN_LAYERS = 1
    MAX_LAYERS = 5
    CHILD_TRAIN_EPOCHS = 15

    start_time = time.time()
    db_name = 'e'
    data, train_data, val_data, test_data = load_and_split_data(db_name)
    if data is None: return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data.to(device);
    val_data.to(device)

    print(f"--- Initializing population (size: {POPULATION_SIZE}) ---")
    population = [create_random_individual(MIN_LAYERS,MAX_LAYERS,CONV_CHOICES,NORM_CHOICES,ACT_CHOICES) for _ in range(POPULATION_SIZE)]

    for gen in range(NUM_GENERATIONS):
        print(f"\n{'=' * 20} 第 {gen + 1}/{NUM_GENERATIONS} 代进化 {'=' * 20}")

        evaluate_population(CHILD_TRAIN_EPOCHS,population, train_data, val_data, device)

        population.sort(key=lambda x: x['fitness'], reverse=True)

        best_fitness = population[0]['fitness']
        avg_fitness = sum(ind['fitness'] for ind in population) / POPULATION_SIZE
        print(f"--- Generation {gen + 1} results: Best fitness = {best_fitness:.4f}, Average fitness = {avg_fitness:.4f} ---")
        next_generation = []

        if ELITISM_COUNT > 0:
            elites = population[:ELITISM_COUNT]
            next_generation.extend(elites)

        while len(next_generation) < POPULATION_SIZE:
            parent1 = tournament_selection(TOURNAMENT_SIZE,population)
            parent2 = tournament_selection(TOURNAMENT_SIZE,population)

            child1, _ = crossover(CROSSOVER_RATE,parent1, parent2)

            mutated_child = mutate(child1,MUTATION_RATE,CONV_CHOICES,NORM_CHOICES,ACT_CHOICES)

            next_generation.append(mutated_child)

        population = next_generation

    print("\n--- Evolution search completed ---")

    print("Performing final evaluation...")
    evaluate_population(population, train_data, val_data, device)
    population.sort(key=lambda x: x['fitness'], reverse=True)

    print("\nFinal best architecture found:")
    best_individual = population[0]
    print(f"  Fitness (Reward): {best_individual['fitness']:.4f}")
    print(f"  Architecture config: {best_individual['arch']}")

    top_k_results = save_top_k_evolutionary(db_name, population, top_k=5)

    end_time = time.time()
    print(f"\nTotal time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()