import os
import time

import torch
import json
import torch_geometric.transforms as T
from bayes_opt import BayesianOptimization


from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel
from Code.search_architecture.lstm_search.Evaluator import LinkPredictionEvaluator
from Code.search_architecture.random_search.run_search import load_and_split_data

CONV_CHOICES = ['sage', 'graphconv', 'transformerconv', 'gatconv', 'gatv2conv']
NORM_CHOICES = ['graphnorm', 'instancenorm', 'layernorm', 'batchnorm', 'identity']
ACT_CHOICES = ['relu', 'tanh', 'leakyrelu', 'identity']

CONV_MAP = {i: choice for i, choice in enumerate(CONV_CHOICES)}
NORM_MAP = {i: choice for i, choice in enumerate(NORM_CHOICES)}
ACT_MAP = {i: choice for i, choice in enumerate(ACT_CHOICES)}


def save_top_k_bayes(db_name, all_results, top_k):
    try:
        script_directory = 'Code/search_architecture/'
        results_dir = os.path.join(script_directory, f'results/{db_name.lower()}')
        print(f"\nEnsuring directory exists: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, f"{db_name}_bayes.json")
        all_results.sort(key=lambda x: x['combined_reward'], reverse=True)
        top_k_results = all_results[:top_k]
        print(f"Preparing to write {len(top_k_results)} records to: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(top_k_results, f, indent=4)
        print(f"Success! Saved the top {len(top_k_results)} architectures to {filepath}")
    except Exception as e:
        print(f"\n[ERROR] An exception occurred while saving the file: {e}")

    return top_k_results



def black_box_function(train_data,val_data,test_data,device,child_train_epochs,**params):

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

        print(f"\nTesting architecture: {layer_configs}")

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

        print(
            f"Result -> AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f}, Combined Reward: {reward:.4f}")

        return reward

    except Exception as e:
        print(f"An error occurred with params. Error: {repr(e)}")
        return -1.0


def main():
    child_train_epochs = 50

    db_name = 'e'
    data, train_data, val_data, test_data = load_and_split_data(db_name)
    if data is None: return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        f=lambda **kwargs: black_box_function(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        device=device,
        child_train_epochs=child_train_epochs,
        **kwargs
    ), pbounds=pbounds, random_state=1, verbose=2)

    start = time.time()
    optimizer.maximize(init_points=50, n_iter=50)
    print(f'used time{time.time()-start}s')
    print("\nBayesian Optimization Finished!")
    print("Best architecture found during the run:")
    print(f"Reward: {optimizer.max['target']:.4f}")

    TOP_K = 60


    all_results_formatted = []
    for i, res in enumerate(optimizer.res):
        params = res['params']
        layer_configs = []
        num_layers = 6

        for layer_idx in range(1, num_layers + 1):
            conv_idx = round(params[f'l{layer_idx}_conv'])
            norm_idx = round(params[f'l{layer_idx}_norm'])
            act_idx = round(params[f'l{layer_idx}_act'])


            conv_idx = max(0, min(conv_idx, len(CONV_MAP) - 1))
            norm_idx = max(0, min(norm_idx, len(NORM_MAP) - 1))
            act_idx = max(0, min(act_idx, len(ACT_MAP) - 1))

            config = {
                'conv': CONV_MAP[conv_idx],
                'norm': NORM_MAP[norm_idx],
                'act': ACT_MAP[act_idx],
            }
            layer_configs.append(config)

        formatted_res = {
            'trial_number': i + 1,
            'combined_reward': res['target'],
            'architecture': layer_configs
        }
        all_results_formatted.append(formatted_res)

    top_k_results = save_top_k_bayes(db_name, all_results_formatted, TOP_K)



if __name__ == "__main__":
    main()
