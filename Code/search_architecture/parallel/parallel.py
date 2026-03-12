import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import ray
import ast
import traceback
import os
import time
import numpy as np
import torch
import json
from bayes_opt import BayesianOptimization, UtilityFunction
from tqdm import tqdm
import pandas as pd

from Code.search_architecture.bayes_search.run_bayes_opt_parallel import evaluate_batch_parallel

from Code.search_architecture.bayes_search.run_bayesian_search import black_box_function, save_top_k_bayes

from Code.search_architecture.evolutionary_search.run_evolutionary_search import create_random_individual, \
    evaluate_population, crossover, mutate, tournament_selection, save_top_k_evolutionary
from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel

from Code.search_architecture.lstm_search.nas_controller import NAS_Controller
from Code.search_architecture.lstm_search.run_nas import load_and_split_data, run_nas_search, full_train_and_evaluate, \
    save_top_k_architectures, to_serializable
from Code.search_architecture.random_search.run_search import run_random_search, save_top_k_random

from Code.search_architecture.random_search.searcher import RandomSearcher


#Reinforcement Learning search by lstm
def lstm_search(db_name,data, train_data, val_data, test_data, device):

    print("lstm search have already started!")
    MAX_LAYERS = 5
    NAS_EPOCHS = 120
    CHILD_TRAIN_EPOCHS = 10
    CONTROLLER_HIDDEN_DIM = 20
    CONTROLLER_LR = 0.001
    TOP_K = 60

    controller = NAS_Controller(max_layers=MAX_LAYERS, hidden_dim=CONTROLLER_HIDDEN_DIM).to(device)
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=CONTROLLER_LR)

    all_results,best_arch,best_arch_reward = run_nas_search(
        controller,
        controller_optimizer,
        train_data,
        val_data,
        nas_epochs=NAS_EPOCHS,
        child_train_epochs=CHILD_TRAIN_EPOCHS,
        device=device,
    )

    lstm_top_k_results = save_top_k_architectures(db_name, all_results, TOP_K)

    # if best_arch:
    #     full_train_and_evaluate(best_arch, data, train_data, val_data, test_data, device)

    return lstm_top_k_results

# bayes search by bayes opt
def bayes_search(db_name, data, train_data, val_data, test_data, device):

    print(f"Parallel Bayes search started for {db_name}!")


    CONV_CHOICES = ['sage', 'graphconv', 'transformerconv', 'gatconv', 'gatv2conv']
    NORM_CHOICES = ['graphnorm', 'instancenorm', 'layernorm', 'batchnorm', 'identity']
    ACT_CHOICES = ['relu', 'tanh', 'leakyrelu', 'identity']

    CONV_MAP = {i: choice for i, choice in enumerate(CONV_CHOICES)}
    NORM_MAP = {i: choice for i, choice in enumerate(NORM_CHOICES)}
    ACT_MAP = {i: choice for i, choice in enumerate(ACT_CHOICES)}


    child_train_epochs = 10
    TOP_K = 60

    batch_size = 4
    num_workers = min(batch_size, os.cpu_count() if os.cpu_count() else 1)


    init_points = 60
    n_iter_points = 60

    print(f"Using {num_workers} workers with batch size {batch_size}")
    print(f"Targeting {init_points} init points and {n_iter_points} opt points.")

    if device == 'cuda':
        print(f"⚠️  Warning: {batch_size} models will share GPU memory")

    pbounds = {
        'l1_conv': (0, len(CONV_MAP) - 0.01), 'l1_norm': (0, len(NORM_MAP) - 0.01), 'l1_act': (0, len(ACT_MAP) - 0.01),
        'l2_conv': (0, len(CONV_MAP) - 0.01), 'l2_norm': (0, len(NORM_MAP) - 0.01), 'l2_act': (0, len(ACT_MAP) - 0.01),
        'l3_conv': (0, len(CONV_MAP) - 0.01), 'l3_norm': (0, len(NORM_MAP) - 0.01), 'l3_act': (0, len(ACT_MAP) - 0.01),
    }

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=1,
        verbose=2,
        allow_duplicate_points=True
    )

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    all_results = []
    start = time.time()

    print("\n" + "=" * 60)
    print("Phase 1: Random Initialization (Parallel)")
    print("=" * 60)

    init_batches = max(1, int(np.ceil(init_points / batch_size)))
    total_init_points_actual = init_batches * batch_size
    print(f"Running {init_batches} init batches (Total approx. {total_init_points_actual} points).")

    for batch_idx in range(init_batches):
        print(f"\n--- Initialization Batch {batch_idx + 1}/{init_batches} ---")
        params_batch = []
        for _ in range(batch_size):
            params = {key: np.random.uniform(bounds[0], bounds[1])
                      for key, bounds in pbounds.items()}
            params_batch.append(params)

        batch_results = evaluate_batch_parallel(
            params_batch, train_data, val_data, test_data,
            device, child_train_epochs, num_workers
        )

        for params, reward, layer_configs in batch_results:
            if layer_configs:
                optimizer.register(params=params, target=reward)
                result_entry = {
                    'trial_number': len(all_results) + 1,
                    'combined_reward': reward,
                    'architecture': layer_configs
                }
                all_results.append(result_entry)
            else:
                print(f"[Warning] Skipping failed evaluation for params.")

    if not optimizer.max:
        print("[Error] Initialization failed to produce any valid results. Stopping.")
        return []
    else:
        print(f"\nInitialization completed. Best so far: {optimizer.max['target']:.4f}")

    print("\n" + "=" * 60)
    print("Phase 2: Bayesian Optimization (Parallel)")
    print("=" * 60)

    n_batches = max(1, int(np.ceil(n_iter_points / batch_size)))
    total_opt_points_actual = n_batches * batch_size
    print(f"Running {n_batches} optimization batches (Total approx. {total_opt_points_actual} points).")

    for batch_idx in range(n_batches):
        print(f"\n--- Optimization Batch {batch_idx + 1}/{n_batches} ---")

        params_batch = []
        for _ in range(batch_size):
            next_point = optimizer.suggest(utility)
            params_batch.append(next_point)

        batch_results = evaluate_batch_parallel(
            params_batch, train_data, val_data, test_data,
            device, child_train_epochs, num_workers
        )

        for params, reward, layer_configs in batch_results:
            if layer_configs:
                optimizer.register(params=params, target=reward)
                result_entry = {
                    'trial_number': len(all_results) + 1,
                    'combined_reward': reward,
                    'architecture': layer_configs
                }
                all_results.append(result_entry)
            else:
                print(f"[Warning] Skipping failed evaluation for params.")

        print(f"Batch {batch_idx + 1} completed. Best so far: {optimizer.max['target']:.4f}")

    elapsed_time = time.time() - start
    print(f'\n{"=" * 60}')
    print(f'Total time: {elapsed_time:.2f}s')
    if all_results:
        print(f'Average time per evaluation: {elapsed_time / len(all_results):.2f}s')
    print(f'{"=" * 60}')

    print("\nBayesian Optimization Finished!")
    if optimizer.max:
        print(f"Best architecture found:")
        print(f"  Reward: {optimizer.max['target']:.4f}")
    else:
        print("No successful evaluations were completed.")

    bayes_top_k_results = save_top_k_bayes(db_name, all_results, TOP_K)

    print(f"\n=== Top-{min(TOP_K, 5)} Architectures (from {len(bayes_top_k_results)} saved) ===")
    for i, res in enumerate(bayes_top_k_results[:5]):
        print(f"\nRank {i + 1}:")
        print(f"  Reward: {res['combined_reward']:.4f}")
        print(f"  Architecture: {res['architecture']}")

    return bayes_top_k_results

# evolutionary search
def evolutionary_search(db_name,data, train_data, val_data, test_data, device):

    print("evolutionary search have already started!")

    CONV_CHOICES = ['sage', 'graphconv', 'transformerconv', 'gatconv', 'gatv2conv']
    NORM_CHOICES = ['graphnorm', 'instancenorm', 'layernorm', 'batchnorm', 'identity']
    ACT_CHOICES = ['relu', 'tanh', 'leakyrelu', 'identity']

    POPULATION_SIZE = 60
    NUM_GENERATIONS = 2
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8
    ELITISM_COUNT = 2
    TOURNAMENT_SIZE = 3
    MIN_LAYERS = 1
    MAX_LAYERS = 5
    CHILD_TRAIN_EPOCHS = 10

    print(f"--- Initializing population (size: {POPULATION_SIZE}) ---")
    population = [create_random_individual(MIN_LAYERS,MAX_LAYERS,CONV_CHOICES,NORM_CHOICES,ACT_CHOICES) for _ in range(POPULATION_SIZE)]

    for gen in range(NUM_GENERATIONS):
        print(f"\n{'=' * 20} Generation {gen + 1}/{NUM_GENERATIONS} {'=' * 20}")

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
    evaluate_population(CHILD_TRAIN_EPOCHS,population, train_data, val_data, device)
    population.sort(key=lambda x: x['fitness'], reverse=True)

    print("\nFinal best architecture found:")
    best_individual = population[0]
    print(f"  Fitness (Reward): {best_individual['fitness']:.4f}")
    print(f"  Architecture config: {best_individual['arch']}")

    evolutionary_top_k_results = save_top_k_evolutionary(db_name, population, top_k=60)

    return evolutionary_top_k_results

# random search
def random_search(db_name,data, train_data, val_data, test_data, device):

    print("random search have already started!")
    MAX_LAYERS = 5
    NAS_EPOCHS = 200
    CHILD_TRAIN_EPOCHS = 10
    TOP_K = 120

    searcher = RandomSearcher(max_layers=MAX_LAYERS)

    all_results,best_arch, best_arch_reward = run_random_search(
        searcher,
        train_data,
        val_data,
        nas_epochs=NAS_EPOCHS,
        child_train_epochs=CHILD_TRAIN_EPOCHS,
        device=device,
    )

    random_top_k_results = save_top_k_random(db_name, all_results, TOP_K)

    return random_top_k_results


def process_architecture_results(results_lists, db_name, output_csv_path):
    print("Loading dataset to retrieve metadata...")
    data, train_data, _, _ = load_and_split_data(db_name)
    if train_data is None:
        raise ValueError("[ERROR] Data loading failed, cannot proceed.")
    print("Data loaded successfully.")

    all_architectures_data = []
    print(f"There are {len(results_lists)} result lists to process...")

    for idx, results_data in enumerate(results_lists, start=1):
        print(f"\nProcessing the {idx} result list, with {len(results_data)} records in total...")
        for result in tqdm(results_data, desc=f"Processing list {idx}"):
            architecture_config = result.get('architecture')
            if not architecture_config:
                continue

            is_valid = all(
                layer_cfg.get(key) is not None for layer_cfg in architecture_config for key in ['conv', 'act', 'norm'])
            if not is_valid:
                continue

            try:
                model = FlexibleGNNModel(architecture_config, train_data.metadata()).to('cpu')

                with torch.no_grad():
                    model.eval()
                    dummy_edge_label_index = train_data['drug', 'protein'].edge_label_index[:, :1]
                    _ = model(train_data.x_dict, train_data.edge_index_dict, dummy_edge_label_index)

                # total_params = count_parameters(model)
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                processed_data = {
                    'combined_reward': result.get('combined_reward', -1.0),
                    'parameters_M': total_params / 1_000_000,
                    'architecture': str(architecture_config),
                    'source_list_index': idx
                }
                all_architectures_data.append(processed_data)

            except Exception as e:
                print("\n" + "!" * 50)
                print(f"❌ Error occurred while processing architecture (List: {idx})")
                print(f"Failed architecture configuration: {architecture_config}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {e}")
                print("-" * 20 + " Full Stack Trace " + "-" * 20)
                traceback.print_exc()
                print("!" * 50 + "\n")

                continue

    if not all_architectures_data:
        print("\nFailed to process any valid architectural data.")
        return None

    results_df = pd.DataFrame(all_architectures_data)
    results_df = results_df.sort_values(by='combined_reward', ascending=False).reset_index(drop=True)

    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_csv_path}")

    print("\n--- All data processing completed ---")
    print(results_df.head())

    return results_df


def calculate_pareto_front(df, reward_col='combined_reward', param_col='parameters_M'):

    df_copy = df.copy()
    df_copy['neg_param'] = -df_copy[param_col]
    points = df_copy[[reward_col, 'neg_param']].values
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            # 如果 j 点在所有指标上 >= i 点，并且至少一个指标 > i 点，则 i 被支配
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False
                break

    return df[is_pareto].copy()


def find_knee_point(pareto_df, reward_col='combined_reward', param_col='parameters_M'):

    if len(pareto_df) < 3:
        print("⚠️ Pareto front has fewer than 3 points. Returning best performance point.")
        return pareto_df.loc[pareto_df[reward_col].idxmax()]

    scaler = MinMaxScaler()
    normalized_points = scaler.fit_transform(pareto_df[[param_col, reward_col]])

    start_point = normalized_points[0]
    end_point = normalized_points[-1]
    line_vec = end_point - start_point
    line_vec_norm = np.sqrt(np.sum(line_vec ** 2))

    if line_vec_norm == 0:
        print("⚠️ Normalized points are co-linear. Returning best performance point.")
        return pareto_df.loc[pareto_df[reward_col].idxmax()]

    vec_from_start = normalized_points - start_point
    cross_product = np.cross(vec_from_start, line_vec)
    distances = np.abs(cross_product) / line_vec_norm

    knee_index = np.argmax(distances)
    return pareto_df.iloc[knee_index]


def analyze_pareto_and_knee(df, db_name, output_plot_image):

    MAIN_POINT_COLOR = '#1E88E5'
    PARETO_COLOR = '#FF6B00'
    KNEE_COLOR = '#FFD300'
    BEST_COLOR = '#00C853'
    LIGHTEST_COLOR = '#E040FB'

    font_axis = {'family': 'Times New Roman', 'size': 16}
    font_tick = {'family': 'Times New Roman', 'size': 13}
    font_legend = {'family': 'Times New Roman', 'size': 12}

    pareto_df = calculate_pareto_front(df)
    pareto_df = pareto_df.sort_values('parameters_M').reset_index(drop=True)
    print(f"🎯 Found {len(pareto_df)} Pareto-optimal solutions")

    knee_point = find_knee_point(pareto_df)
    print(f"🏆 Found the best trade-off point (Knee Point)")

    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)

    ax.scatter(df['parameters_M'], df['combined_reward'],
               color=MAIN_POINT_COLOR, alpha=0.50, s=80,
               label='All Candidate Architectures')

    ax.plot(pareto_df['parameters_M'], pareto_df['combined_reward'],
            color=PARETO_COLOR, marker='o', linestyle='--',
            linewidth=2, markersize=10,
            label='Pareto Front')


    key_points_info = {
        'Best Trade-off (Knee Point)': (knee_point, KNEE_COLOR, 250, '*'),
        'Best Performance': (pareto_df.loc[pareto_df['combined_reward'].idxmax()], BEST_COLOR, 140, 'D'),
        'Most Lightweight': (pareto_df.loc[pareto_df['parameters_M'].idxmin()], LIGHTEST_COLOR, 140, 's')
    }

    for label, (point_data, color, size, marker) in key_points_info.items():
        ax.scatter(point_data['parameters_M'], point_data['combined_reward'],
                   color=color, s=size, marker=marker,
                   edgecolor='black', linewidth=0.6, zorder=5, label=label)

    ax.annotate(
        f"Knee Point\nReward: {knee_point['combined_reward']:.4f}\nParams: {knee_point['parameters_M']:.3f}M",
        xy=(knee_point['parameters_M'], knee_point['combined_reward']),
        xytext=(40, -40), textcoords='offset points',
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85),
        arrowprops=dict(arrowstyle='->', color='black')
    )

    ax.set_xlabel('Model Parameters (M)', fontdict=font_axis)
    ax.set_ylabel('Comprehensive Validation Performance Score', fontdict=font_axis)

    ax.tick_params(labelsize=font_tick['size'])
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')

    ax.grid(linestyle="--", color='lightgray', linewidth=0.8)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    ax.legend(prop=font_legend,
              frameon=True, facecolor='white',
              edgecolor='gray', framealpha=0.85,
              loc='lower right')

    plt.tight_layout()

    if output_plot_image:
        png_path = f"{output_plot_image}/{db_name.lower()}_pareto_front.png"
        pdf_path = f"{output_plot_image}/{db_name.lower()}_pareto_front.pdf"

        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')

        print(f"📈 Saved PNG: {png_path}")
        print(f"📘 Saved PDF: {pdf_path}")
    else:
        plt.show()

    print("\n" + "=" * 80)
    print("🏆 Final Architecture Selection (based on Knee Point):")
    print(f"  - Combined Reward: {knee_point['combined_reward']:.6f}")
    print(f"  - Parameters (M): {knee_point['parameters_M']:.6f}")
    print(f"  - Architecture Definition: {knee_point['architecture']}")
    if 'source_file' in knee_point:
        print(f"  - Source File: {knee_point['source_file']}")
    print("=" * 80)

    return pareto_df, knee_point


@ray.remote(num_gpus=0.25)
def lstm_search_remote(db_name,data, train_data, val_data, test_data, device):
    return lstm_search(db_name,data, train_data, val_data, test_data, device)

@ray.remote(num_gpus=0.25)
def bayes_search_remote(db_name,data, train_data, val_data, test_data, device):
    return bayes_search(db_name,data, train_data, val_data, test_data, device)

@ray.remote(num_gpus=0.25)
def evolutionary_search_remote(db_name,data, train_data, val_data, test_data, device):
    return evolutionary_search(db_name,data, train_data, val_data, test_data, device)

@ray.remote(num_gpus=0.25)
def random_search_remote(db_name,data, train_data, val_data, test_data, device):
    return random_search(db_name,data, train_data, val_data, test_data, device)

def main():
    ray.init(num_gpus=1)
    data_list = ['drugbank','biosnap','bindingdb','davis','e','gpcr','ic','nr']#'drugbank','biosnap','bindingdb','davis','e','gpcr','ic','nr'
    # all_results = []
    for db_name in data_list:
        print(f"The program runs on the {db_name} dataset.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        output_path = f'Code/search_architecture/results_6/{db_name.lower()}'
        os.makedirs(output_path, exist_ok=True)
        print("Initializing Ray...")

        print(f"\n{'=' * 25} Processing Dataset: {db_name.upper()} {'=' * 25}\n")

        data, train_data, val_data, test_data = load_and_split_data(db_name)
        if data is None:
            print(f"Skipping dataset {db_name} due to missing file.")

        train_data.to(device)
        val_data.to(device)
        test_data.to(device)
        print(f"Using device: {device} for dataset {db_name.upper()}")

        start_parallel = time.time()

        print("Starting all parallel tasks (and passing parameters)...")
        futures = [
            lstm_search_remote.remote(db_name,data, train_data, val_data, test_data, device),
            bayes_search_remote.remote(db_name,data, train_data, val_data, test_data, device),
            evolutionary_search_remote.remote(db_name,data, train_data, val_data, test_data, device),
            random_search_remote.remote(db_name,data, train_data, val_data, test_data, device)
        ]

        print("All tasks have been launched in the background.")

        results_parallel = ray.get(futures)
        med_time = time.time()
        print(f"Search architecture execution time: {med_time - start_parallel} seconds")

        df = process_architecture_results(
            results_parallel,
            db_name=db_name,
            output_csv_path=os.path.join(output_path,f'{db_name.lower()}_results.csv'),
        )

        pareto_df, knee_point = analyze_pareto_and_knee(
            df,
            db_name=db_name,
            output_plot_image=output_path
        )

        architecture = ast.literal_eval(knee_point["architecture"])
        result_dict = full_train_and_evaluate(architecture, data, train_data, val_data, test_data, device)

        end_parallel = time.time()
        duration = end_parallel - start_parallel

        print(f"\nRay total execution time: {duration:.2f} seconds")
        print(f"Pareto front key point identification time: {end_parallel - med_time} seconds")

        result_dict["architecture"] = architecture
        result_dict["used time"] = duration
        result_dict["search used time"] = med_time - start_parallel
        result_dict["Pareto used time"] = end_parallel - med_time

        result_dict = to_serializable(result_dict)

        with open(os.path.join(output_path, f'{db_name.lower()}_log.json'), "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print("finish!!!")

if __name__ == "__main__":
    main()

