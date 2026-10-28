import os
import torch
import json
import torch_geometric.transforms as T
from Code.search_architecture.lstm_search.Evaluator import LinkPredictionEvaluator
from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel
from Code.search_architecture.random_search.searcher import RandomSearcher


def save_top_k_random(db_name, all_results, top_k):

    script_directory = 'Code/search_architecture/'
    results_dir = os.path.join(script_directory, f'results/{db_name.lower()}')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{db_name}_random.json")

    all_results.sort(key=lambda x: x['combined_reward'], reverse=True)

    top_k_results = all_results[:top_k]

    with open(filepath, 'w') as f:
        json.dump(top_k_results, f, indent=4)
    print(f"  Saved the top {len(top_k_results)} best architectures to {filepath}")
    return top_k_results


def run_random_search(searcher, train_data, val_data, nas_epochs, child_train_epochs,device):

    reward_weights = {'AUPR': 0.5, 'AUC': 0.5}

    print(f"Using multi-objective reward with weights: {reward_weights}")
    print("Starting random architecture search...")
    best_reward = -1
    best_architecture = None
    all_results = []

    for epoch in range(nas_epochs):
        layer_configs = searcher.sample()

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
            current_arch_data = {
                'epoch': epoch + 1,
                'combined_reward': reward,
                'metrics': {
                    'AUC': val_metrics['AUC'],
                    'AUPR': val_metrics['AUPR']
                },
                'architecture': layer_configs
            }
            all_results.append(current_arch_data)

            if reward > best_reward:
                best_reward = reward
                best_architecture = layer_configs
                print(f"New best architecture found! Comprehensive reward: {best_reward:.4f} (at epoch {epoch + 1})")
                print(f" -> Metrics: AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f}")
                print(f" -> Architecture: {best_architecture}")
            if (epoch + 1) % 10 == 0:
                print(f"Random search {epoch + 1}/{nas_epochs} Current reward: {reward:.4f}Bestreward: {best_reward: .4f}")

        except Exception as e:
            print(f"Architecture error: {layer_configs}. Skipped. Error: {e}")
    print("\nSearch completed!")

    print(f"Best architecture found (comprehensive reward = {best_reward:.4f}):")
    print(best_architecture)
    return all_results,best_architecture, best_reward


def load_and_split_data(database_name):
    PATH_DATA = os.path.join('Data', database_name.upper(), f'hetero_data_{database_name}.pt')

    print(f"Loading data from path: {PATH_DATA}")
    if not os.path.exists(PATH_DATA):
        print(f"Data file not found: {PATH_DATA}")
        return None, None, None, None

    data = torch.load(PATH_DATA)
    data = T.ToUndirected()(data)
    del data['protein', 'rev_interaction', 'drug'].edge_label
    split = T.RandomLinkSplit(
        num_val=0.1, num_test=0.2, is_undirected=True,
        add_negative_train_samples=True, neg_sampling_ratio=1.0,
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
    for epoch in range(500):
        evaluator.train_one_epoch(train_data)
        if epoch % 50 == 0:
            val_metrics = evaluator.evaluate(val_data)
            print(f"Full training epoch {epoch:03d} | Validation AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f}")
    test_metrics = evaluator.evaluate(test_data)
    print("\n--- Final test results ---")
    print(
        f"Test AUC: {test_metrics['AUC']:.4f}, AUPR: {test_metrics['AUPR']:.4f}, ACC: {test_metrics['ACC']:.4f}, F1: {test_metrics['F1']:.4f}")

def main():
    DATABASE_LIST = ['e']
    for db_name in DATABASE_LIST:
        print(f"\n{'=' * 25} Processing dataset: {db_name.upper()} {'=' * 25}\n")
        data, train_data, val_data, test_data = load_and_split_data(db_name)
        if data is None:
            print(f"Skipped dataset {db_name} due to missing file.")
            continue
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_data.to(device)
        val_data.to(device)
        test_data.to(device)
        print(f"Using device {device} for dataset {db_name.upper()}.")

        MAX_LAYERS = 6
        NAS_EPOCHS = 200
        CHILD_TRAIN_EPOCHS = 50
        TOP_K = 60

        searcher = RandomSearcher(max_layers=MAX_LAYERS)

        all_results,best_arch, best_arch_reward = run_random_search(
            searcher,
            train_data,
            val_data,
            nas_epochs=NAS_EPOCHS,
            child_train_epochs=CHILD_TRAIN_EPOCHS,
            device=device,
        )

        top_k_results = save_top_k_random(db_name, all_results, TOP_K)

        if best_arch:
            full_train_and_evaluate(best_arch, data, train_data, val_data, test_data, device)


if __name__ == "__main__":
    main()
