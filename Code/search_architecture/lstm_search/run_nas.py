import os

import numpy as np
import torch
import json
import sys
import io
import json
import torch_geometric.transforms as T
from Code.search_architecture.lstm_search.Evaluator import LinkPredictionEvaluator
from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel
from Code.search_architecture.lstm_search.nas_controller import NAS_Controller


def save_top_k_architectures(db_name, all_results, top_k):

    try:
        script_directory = f'Code/search_architecture'
        results_dir = os.path.join(script_directory, f'results/{db_name.lower()}')

        print(f"\nEnsuring directory exists: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)

        filepath = os.path.join(results_dir, f"{db_name}_lstm.json")

        all_results.sort(key=lambda x: x['combined_reward'], reverse=True)
        top_k_results = all_results[:top_k]

        print(f"Preparing to write {len(top_k_results)} records to: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(top_k_results, f, indent=4)

        print(f"Success! Saved the top {len(top_k_results)} best architectures to {filepath}")

        return top_k_results
    except Exception as e:
        print(f"\n[ERROR] An exception occurred while saving the file: {e}")

def run_nas_search(controller, controller_optimizer, train_data, val_data, nas_epochs, child_train_epochs, device):

    reward_weights = {'AUPR': 0.5, 'AUC': 0.5}
    print(f"Using multi-metric reward with weights: {reward_weights}")

    print("Starting Neural Architecture Search...")
    baseline = 0.0
    moving_avg_decay = 0.95
    best_reward = -1
    best_architecture = None
    all_results = []

    for epoch in range(nas_epochs):
        layer_configs, log_prob = controller.sample()
        try:
            child_model = FlexibleGNNModel(layer_configs, train_data.metadata()).to(device)
            child_optimizer = torch.optim.Adam(child_model.parameters(), lr=0.01)
            criterion = torch.nn.BCEWithLogitsLoss()
            evaluator = LinkPredictionEvaluator(child_model, criterion, child_optimizer, device)

            for _ in range(child_train_epochs):
                evaluator.train_one_epoch(train_data)

            val_metrics = evaluator.evaluate(val_data)
            reward = (reward_weights['AUPR'] * val_metrics['AUPR'] +
                      reward_weights['AUC'] * val_metrics['AUC'])

            baseline = moving_avg_decay * baseline + (1 - moving_avg_decay) * reward
            policy_loss = -log_prob * (reward - baseline)
            controller_optimizer.zero_grad()
            policy_loss.backward()
            controller_optimizer.step()


            arch_data_to_save = {
                'epoch': epoch + 1,
                'combined_reward': reward,
                'metrics': {
                    'AUC': val_metrics['AUC'],
                    'AUPR': val_metrics['AUPR']
                },
                'architecture': layer_configs
            }
            all_results.append(arch_data_to_save)

            if reward > best_reward:
                best_reward = reward
                best_architecture = layer_configs
                print(f"New Best Architecture Found! Combined Reward: {best_reward:.4f}")
                print(f"Metrics -> AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f}")
                print(f"Arch: {best_architecture}")

            if (epoch + 1) % 10 == 0:
                print(
                    f"NAS Epoch {epoch + 1}/{nas_epochs} | Last Reward: {reward:.4f} | Baseline: {baseline:.4f} | Policy Loss: {policy_loss.item():.4f}")

        except Exception as e:
            print(f"Error occurred with architecture: {layer_configs}. Skipping. Error: {e}")
            policy_loss = -log_prob * (-1.0 - baseline)
            controller_optimizer.zero_grad()
            policy_loss.backward()
            controller_optimizer.step()

    print("\nNAS Search Finished!")

    print(f"Best architecture found during the run with Combined Reward = {best_reward:.4f}:")
    print(best_architecture)
    return all_results, best_architecture, best_reward

def load_and_split_data(database_name):
    PATH_DATA = os.path.join('Data', database_name.upper(), f'hetero_data_{database_name}.pt')
    print(f"Loading data from: {PATH_DATA}")
    if not os.path.exists(PATH_DATA):
        print(f"Data file not found: {PATH_DATA}")
        return None, None, None, None
    data = torch.load(PATH_DATA)
    data = T.ToUndirected()(data)
    del data['protein', 'rev_interaction', 'drug'].edge_label
    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        disjoint_train_ratio=0.3,
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
    )
    train_data, val_data, test_data = split(data)
    return data, train_data, val_data, test_data


def full_train_and_evaluate(best_arch, data, train_data, val_data, test_data, device):
    print("\n--- Starting full training with the best architecture ---")

    model = FlexibleGNNModel(best_arch, data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    evaluator = LinkPredictionEvaluator(model, criterion, optimizer, device)

    max_test_auc = 0
    max_test_aupr = 0
    final_test_metrics = {}

    for epoch in range(500):
        evaluator.train_one_epoch(train_data)
        val_metrics = evaluator.evaluate(val_data)
        test_metrics = evaluator.evaluate(test_data)

        if test_metrics['AUC'] > max_test_auc:
            max_test_auc = test_metrics['AUC']
        if test_metrics['AUPR'] > max_test_aupr:
            max_test_aupr = test_metrics['AUPR']

        if epoch % 50 == 0:
            print(f"Train Epoch {epoch:03d} | "
                  f"Val AUC: {val_metrics['AUC']:.4f}, Val AUPR: {val_metrics['AUPR']:.4f} | "
                  f"Test AUC: {test_metrics['AUC']:.4f}, Test AUPR: {test_metrics['AUPR']:.4f}")

        final_test_metrics = test_metrics

    print("--- Final Test Results ---")
    print(f"Test AUC: {final_test_metrics['AUC']:.4f}, "
          f"AUPR: {final_test_metrics['AUPR']:.4f}, "
          f"ACC: {final_test_metrics['ACC']:.4f}, "
          f"F1: {final_test_metrics['F1']:.4f}")
    print(f"MAX Test AUC: {max_test_auc:.4f}, MAX Test AUPR: {max_test_aupr:.4f}\n")


    result_dict = {
        "final_auc": final_test_metrics['AUC'],
        "final_aupr": final_test_metrics['AUPR'],
        "final_loss": final_test_metrics['Loss'],
        "final_acc": final_test_metrics['ACC'],
        "final_f1": final_test_metrics['F1'],
        "max_test_auc": max_test_auc,
        "max_test_aupr": max_test_aupr,
    }

    return to_serializable(result_dict)

def to_serializable(obj):

    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int64, np.int32)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj

def main():
    DATABASE_LIST = ['nr']

    for db_name in DATABASE_LIST:
        print(f"\n{'=' * 25} Processing Dataset: {db_name.upper()} {'=' * 25}\n")

        data, train_data, val_data, test_data = load_and_split_data(db_name)
        if data is None:
            print(f"Skipping dataset {db_name} due to missing file.")
            continue

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_data.to(device)
        val_data.to(device)
        test_data.to(device)
        print(f"Using device: {device} for dataset {db_name.upper()}")

        MAX_LAYERS = 6
        NAS_EPOCHS = 120
        CHILD_TRAIN_EPOCHS = 50
        CONTROLLER_HIDDEN_DIM = 20
        CONTROLLER_LR = 0.001
        TOP_K = 60

        controller = NAS_Controller(max_layers=MAX_LAYERS, hidden_dim=CONTROLLER_HIDDEN_DIM).to(device)
        controller_optimizer = torch.optim.Adam(controller.parameters(), lr=CONTROLLER_LR)

        all_results,best_arch, best_arch_reward = run_nas_search(
            controller,
            controller_optimizer,
            train_data,
            val_data,
            nas_epochs=NAS_EPOCHS,
            child_train_epochs=CHILD_TRAIN_EPOCHS,
            device=device,
        )

        top_k_results = save_top_k_architectures(db_name, all_results, TOP_K)

        if best_arch:
            full_train_and_evaluate(best_arch, data, train_data, val_data, test_data, device)


if __name__ == "__main__":
    main()
