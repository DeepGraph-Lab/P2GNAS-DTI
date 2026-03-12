
import os
import time
import json
import random
import numpy as np
import torch


try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
    from botorch.acquisition import qExpectedImprovement, UpperConfidenceBound, ExpectedImprovement
except Exception as e:
    raise ImportError("Please install botorch and gpytorch first:\n    pip install botorch gpytorch\nOriginal error: " + repr(e))

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
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, f"{db_name}_bayes.json")
        all_results.sort(key=lambda x: x['combined_reward'], reverse=True)
        top_k_results = all_results[:top_k]
        with open(filepath, 'w') as f:
            json.dump(top_k_results, f, indent=4)
        print(f"Success! Saved the top {len(top_k_results)} best architectures to {filepath}")
    except Exception as e:
        print(f"\n[ERROR] An exception occurred while saving the file: {e}")
        top_k_results = []

    return top_k_results


def params_to_onehot(params, num_layers=3):

    def one_hot(idx, size):
        v = np.zeros(size, dtype=np.float32)
        i = int(round(float(idx)))
        i = max(0, min(i, size - 1))
        v[i] = 1.0
        return v

    feats = []
    for i in range(1, num_layers + 1):
        feats.append(one_hot(params[f"l{i}_conv"], len(CONV_CHOICES)))
        feats.append(one_hot(params[f"l{i}_norm"], len(NORM_CHOICES)))
        feats.append(one_hot(params[f"l{i}_act"], len(ACT_CHOICES)))
    return np.concatenate(feats)


def params_to_layer_configs(params, num_layers=3):
    layer_configs = []
    for i in range(1, num_layers + 1):
        conv_idx = int(round(params[f'l{i}_conv']))
        norm_idx = int(round(params[f'l{i}_norm']))
        act_idx = int(round(params[f'l{i}_act']))
        conv_idx = max(0, min(conv_idx, len(CONV_MAP) - 1))
        norm_idx = max(0, min(norm_idx, len(NORM_MAP) - 1))
        act_idx = max(0, min(act_idx, len(ACT_MAP) - 1))
        layer_configs.append({
            'conv': CONV_MAP[conv_idx],
            'norm': NORM_MAP[norm_idx],
            'act': ACT_MAP[act_idx],
        })
    return layer_configs


def random_sample_params():
    p = {}
    for l in (1, 2, 3):
        p[f'l{l}_conv'] = int(np.random.randint(0, len(CONV_CHOICES)))
        p[f'l{l}_norm'] = int(np.random.randint(0, len(NORM_CHOICES)))
        p[f'l{l}_act'] = int(np.random.randint(0, len(ACT_CHOICES)))
    return p


def black_box_function(train_data, val_data, test_data, device, child_train_epochs, params):

    try:
        layer_configs = params_to_layer_configs(params, num_layers=3)
        print(f"\n[Eval] Testing architecture: {layer_configs}")

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

        print(f"[Eval] Result -> AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f}, Reward: {reward:.4f}")

        try:
            del evaluator
            del child_optimizer
            del child_model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(reward)
    except Exception as e:
        print(f"[black_box_function error] {repr(e)}")
        return float(-1e3)


def main():

    child_train_epochs = 15
    db_name = 'e'
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data, train_data, val_data, test_data = load_and_split_data(db_name)
    if data is None:
        print("load_and_split_data returned None, exiting.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    init_points = 5
    n_iter = 20
    candidate_pool_size = 3000
    top_k_save = 5

    X_list = []
    Y_list = []

    start_time = time.time()
    print(f"[Main] Initializing {init_points} random points for real evaluation...")

    param_history = []
    X_list = []
    Y_list = []
    all_results = []

    for i in range(init_points):
        p = random_sample_params()
        y = black_box_function(train_data, val_data, test_data, device, child_train_epochs, p)
        X_list.append(params_to_onehot(p))
        Y_list.append([y])
        param_history.append(p)
        all_results.append({
            'trial_number': len(param_history),
            'combined_reward': float(y),
            'architecture': params_to_layer_configs(p)
        })

    X = torch.from_numpy(np.vstack(X_list)).to(device=device, dtype=torch.float32)
    Y = torch.from_numpy(np.vstack(Y_list)).to(device=device, dtype=torch.float32)
    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0)
    y_std = torch.where(y_std == 0, torch.tensor(1.0, device=device, dtype=y_std.dtype), y_std)
    Y_norm = (Y - y_mean) / y_std

    for it in range(n_iter):
        print(f"\n[BO] Iteration {it+1}/{n_iter} - training GP on device {device} with {X.shape[0]} observations")
        X_train = X.double()
        Y_train = Y_norm.double()

        gp = SingleTaskGP(X_train, Y_train)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_model(mll)
        except Exception as e:
            print(f"[GP fit error] {e}. 尝试继续（用当前 GP）")

        best_f = Y_train.max().item()
        acq = ExpectedImprovement(model=gp, best_f=best_f)

        candidates = []
        cand_onehots = []
        for _ in range(candidate_pool_size):
            c = random_sample_params()
            candidates.append(c)
            cand_onehots.append(params_to_onehot(c))

        candidate_x = torch.from_numpy(np.vstack(cand_onehots)).to(device=device, dtype=torch.float32).double()


        with torch.no_grad():

            try:
                acq_vals = acq(candidate_x.unsqueeze(1))
            except Exception as e:

                try:
                    acq_vals = acq(candidate_x.float().unsqueeze(1))
                except Exception as e2:
                    print(f"[acq eval error] primary error: {e}; fallback error: {e2}")
                    raise


            acq_vals_cpu = acq_vals.squeeze(-1).cpu()
            best_idx = int(torch.argmax(acq_vals_cpu).item())

        best_candidate = candidates[best_idx]
        print(f"[BO] 选中 candidate index {best_idx} from pool. Params: {best_candidate}")

        y_new = black_box_function(train_data, val_data, test_data, device, child_train_epochs, best_candidate)

        x_new = torch.from_numpy(params_to_onehot(best_candidate)).to(device=device, dtype=torch.float32).unsqueeze(0)
        X = torch.cat([X, x_new], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_new]], device=device, dtype=torch.float32)], dim=0)

        y_mean = Y.mean(dim=0)
        y_std = Y.std(dim=0)
        y_std = torch.where(y_std == 0, torch.tensor(1.0, device=device, dtype=y_std.dtype), y_std)
        Y_norm = (Y - y_mean) / y_std

        param_history.append(best_candidate)
        all_results.append({
            'trial_number': len(param_history),
            'combined_reward': float(y_new),
            'architecture': params_to_layer_configs(best_candidate)
        })

    total_used = time.time() - start_time
    print(f"\nSearch completed in {total_used:.1f}s, saving Top-{top_k_save} ...")

    top_k_results = save_top_k_bayes(db_name, all_results, top_k_save)

    if top_k_results:
        best = top_k_results[0]
        best_arch = best['architecture']
        print("[Final Eval] Performing final training and evaluation on the best architecture using the test set...")
        try:
            best_model = FlexibleGNNModel(best_arch, train_data.metadata()).to(device)
            best_optimizer = torch.optim.Adam(best_model.parameters(), lr=0.01)
            criterion = torch.nn.BCEWithLogitsLoss()
            evaluator = LinkPredictionEvaluator(best_model, criterion, best_optimizer, device)
            for _ in range(max(10, child_train_epochs)):
                evaluator.train_one_epoch(train_data)
            test_metrics = evaluator.evaluate(test_data)
            best['test_metrics'] = test_metrics
            script_directory = 'Code/search_architecture/'
            filepath = os.path.join(script_directory, f'results/{db_name.lower()}/{db_name}_bayes.json')
            with open(filepath, 'w') as f:
                json.dump(top_k_results, f, indent=4)
            print(f"[Final Eval] Test -> AUC: {test_metrics['AUC']:.4f}, AUPR: {test_metrics['AUPR']:.4f}")
        except Exception as e:
            print(f"[Final Eval Error] {e}")

    print("All done.")


if __name__ == "__main__":
    main()