import os
import time
import numpy as np
import torch
import json
from bayes_opt import BayesianOptimization, UtilityFunction
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 复用您项目中的模型和评估器
from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel
from Code.search_architecture.lstm_search.Evaluator import LinkPredictionEvaluator
from Code.search_architecture.random_search.run_search import load_and_split_data

CONV_CHOICES = ['sage', 'graphconv', 'transformerconv', 'gatconv', 'gatv2conv']
NORM_CHOICES = ['graphnorm', 'instancenorm', 'layernorm', 'batchnorm', 'identity']
ACT_CHOICES = ['relu', 'tanh', 'leakyrelu', 'identity']

CONV_MAP = {i: choice for i, choice in enumerate(CONV_CHOICES)}
NORM_MAP = {i: choice for i, choice in enumerate(NORM_CHOICES)}
ACT_MAP = {i: choice for i, choice in enumerate(ACT_CHOICES)}

result_lock = threading.Lock()


def save_top_k_bayes(db_name, all_results, top_k):
    try:
        script_directory = 'Code/search_architecture/'
        results_dir = os.path.join(script_directory, f'results/{db_name.lower()}')
        print(f"\nEnsuring directory exists: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, f"{db_name}_bayes_parallel.json")

        all_results.sort(key=lambda x: x['combined_reward'], reverse=True)
        top_k_results = all_results[:top_k]

        print(f"Preparing to write {len(top_k_results)} records to: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(top_k_results, f, indent=4)
        print(f"Success! Saved top {len(top_k_results)} architectures to {filepath}")
    except Exception as e:
        print(f"\n[ERROR] An exception occurred while saving the file: {e}")

    return top_k_results


def black_box_function(train_data, val_data, test_data, device, child_train_epochs, thread_id, **params):

    try:
        layer_configs = []
        num_layers = 6

        for i in range(1, num_layers + 1):
            conv_idx = round(params[f'l{i}_conv'])
            norm_idx = round(params[f'l{i}_norm'])
            act_idx = round(params[f'l{i}_act'])

            conv_idx = max(0, min(conv_idx, len(CONV_MAP) - 1))
            norm_idx = max(0, min(norm_idx, len(NORM_MAP) - 1))
            act_idx = max(0, min(act_idx, len(ACT_MAP) - 1))

            config = {
                'conv': CONV_MAP[conv_idx],
                'norm': NORM_MAP[norm_idx],
                'act': ACT_MAP[act_idx],
            }
            layer_configs.append(config)

        print(f"[Thread-{thread_id}] Testing architecture: {layer_configs}")

        child_model = FlexibleGNNModel(layer_configs, train_data.metadata()).to(device)
        child_optimizer = torch.optim.Adam(child_model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        evaluator = LinkPredictionEvaluator(child_model, criterion, child_optimizer, device)

        for _ in range(child_train_epochs):
            evaluator.train_one_epoch(train_data)

        val_metrics = evaluator.evaluate(val_data)
        reward_weights = {'AUPR': 0.5, 'AUC': 0.5}
        reward = (reward_weights['AUPR'] * val_metrics['AUPR'] +
                  reward_weights['AUC'] * val_metrics['AUC'])

        print(f"[Thread-{thread_id}] Result -> AUC: {val_metrics['AUC']:.4f}, "
              f"AUPR: {val_metrics['AUPR']:.4f}, Combined Reward: {reward:.4f}")

        return reward, layer_configs

    except Exception as e:
        print(f"[Thread-{thread_id}] Error: {repr(e)}")
        return -1.0, []


def evaluate_batch_parallel(params_batch, train_data, val_data, test_data, device,
                            child_train_epochs, num_workers):

    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_params = {
            executor.submit(
                black_box_function,
                train_data, val_data, test_data, device, child_train_epochs, i, **params
            ): (i, params)
            for i, params in enumerate(params_batch)
        }

        for future in as_completed(future_to_params):
            thread_id, params = future_to_params[future]
            try:
                reward, layer_configs = future.result()
                results.append((params, reward, layer_configs))
            except Exception as exc:
                print(f'[Thread-{thread_id}] generated an exception: {exc}')
                results.append((params, -1.0, []))

    return results


def main():
    child_train_epochs = 50

    batch_size = 2
    num_workers = os.cpu_count()
    print(f"Using {num_workers} worker threads")

    db_name = 'e'
    data, train_data, val_data, test_data = load_and_split_data(db_name)
    if data is None:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"⚠️  Warning: {batch_size} models will share GPU memory")
        print(f"   If OOM occurs, reduce batch_size or use device='cpu'")

    train_data.to(device)
    val_data.to(device)

    pbounds = {
        'l1_conv': (0, len(CONV_MAP) - 0.01), 'l1_norm': (0, len(NORM_MAP) - 0.01), 'l1_act': (0, len(ACT_MAP) - 0.01),
        'l2_conv': (0, len(CONV_MAP) - 0.01), 'l2_norm': (0, len(NORM_MAP) - 0.01), 'l2_act': (0, len(ACT_MAP) - 0.01),
        'l3_conv': (0, len(CONV_MAP) - 0.01), 'l3_norm': (0, len(NORM_MAP) - 0.01), 'l3_act': (0, len(ACT_MAP) - 0.01),
        'l4_conv': (0, len(CONV_MAP) - 0.01), 'l4_norm': (0, len(NORM_MAP) - 0.01), 'l4_act': (0, len(ACT_MAP) - 0.01),
        'l5_conv': (0, len(CONV_MAP) - 0.01), 'l5_norm': (0, len(NORM_MAP) - 0.01), 'l5_act': (0, len(ACT_MAP) - 0.01),
        'l6_conv': (0, len(CONV_MAP) - 0.01), 'l6_norm': (0, len(NORM_MAP) - 0.01), 'l6_act': (0, len(ACT_MAP) - 0.01),
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

    init_points = 50
    init_batches = init_points // batch_size

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
            optimizer.register(params=params, target=reward)

            result_entry = {
                'trial_number': len(all_results) + 1,
                'combined_reward': reward,
                'architecture': layer_configs
            }
            all_results.append(result_entry)

    print(f"\nInitialization completed. Best so far: {optimizer.max['target']:.4f}")

    print("\n" + "=" * 60)
    print("Phase 2: Bayesian Optimization (Parallel)")
    print("=" * 60)

    n_iter = 50
    n_batches = n_iter // batch_size

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
            optimizer.register(params=params, target=reward)

            result_entry = {
                'trial_number': len(all_results) + 1,
                'combined_reward': reward,
                'architecture': layer_configs
            }
            all_results.append(result_entry)

        print(f"Batch {batch_idx + 1} completed. Best so far: {optimizer.max['target']:.4f}")

    elapsed_time = time.time() - start
    print(f'\n{"=" * 60}')
    print(f'Total time: {elapsed_time:.2f}s')
    print(f'Average time per evaluation: {elapsed_time / len(all_results):.2f}s')
    print(f'{"=" * 60}')

    print("\nBayesian Optimization Finished!")
    print(f"Best architecture found:")
    print(f"  Reward: {optimizer.max['target']:.4f}")
    print(f"  Parameters: {optimizer.max['params']}")

    TOP_K = 60
    top_k_results = save_top_k_bayes(db_name, all_results, TOP_K)

    print("\n=== Top-5 Architectures ===")
    for i, res in enumerate(top_k_results):
        print(f"\nRank {i + 1}:")
        print(f"  Reward: {res['combined_reward']:.4f}")
        print(f"  Architecture: {res['architecture']}")


if __name__ == "__main__":
    main()
