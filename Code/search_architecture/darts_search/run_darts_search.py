import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv, GraphConv, TransformerConv, GATConv, GATv2Conv
)
from torch_geometric import transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from torch_geometric.data import HeteroData


CONV_OPS = {
    'sage': SAGEConv,
    'graphconv': GraphConv,
    'transformerconv': TransformerConv,
    'gatconv': GATConv,
    'gatv2conv': GATv2Conv,
}


class EdgeClassifier(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        drug_vec = z_dict['drug'][row]
        protein_vec = z_dict['protein'][col]
        interaction_vec = drug_vec * protein_vec
        z = torch.cat([drug_vec, protein_vec, interaction_vec], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class DARTSHeteroLayer(nn.Module):
    def __init__(self, metadata, hidden_dim=60, heads=4):
        super().__init__()
        node_types, edge_types = metadata
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.ops = nn.ModuleDict()
        for e in self.edge_types:
            key = "__".join(e)
            conv_dict = nn.ModuleDict()
            for name, Op in CONV_OPS.items():
                n = name.lower()
                if n in ['gatconv', 'gatv2conv']:
                    conv = Op((-1, -1), hidden_dim // heads,
                              heads=heads, add_self_loops=False)
                elif n == 'transformerconv':
                    conv = Op((-1, -1), hidden_dim // heads, heads=heads)
                else:
                    conv = Op((-1, -1), hidden_dim)
                conv_dict[n] = conv
            self.ops[key] = conv_dict

        for e in self.edge_types:
            key = "__".join(e)
            self.register_parameter(
                f"alpha__{key}", nn.Parameter(torch.randn(len(CONV_OPS)))
            )

    def forward(self, x_dict, edge_index_dict):
        device = next(self.parameters()).device
        out_dict = {nt: torch.zeros((x_dict[nt].size(0), self.hidden_dim),
                                    device=device) for nt in x_dict.keys()}

        for e in self.edge_types:
            key = "__".join(e)
            src, rel, dst = e
            edge_index = edge_index_dict[(src, rel, dst)]
            conv_dict = self.ops[key]
            alpha = getattr(self, f"alpha__{key}")
            weights = F.softmax(alpha, dim=-1)

            x_src, x_dst = x_dict[src], x_dict[dst]
            out = 0
            for i, (op_name, op) in enumerate(conv_dict.items()):
                try:
                    out_op = op((x_src, x_dst), edge_index)
                except TypeError:
                    out_op = op(x_src, edge_index)
                out = out + weights[i] * out_op

            out_dict[dst] = out_dict[dst] + out

        return out_dict


class DARTSHeteroModel(nn.Module):
    def __init__(self, metadata, hidden_dim=60, num_layers=3, heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            DARTSHeteroLayer(metadata, hidden_dim, heads)
            for _ in range(num_layers)
        ])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.decoder = EdgeClassifier(hidden_dim)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        for i, layer in enumerate(self.layers):
            out_dict = layer(x_dict, edge_index_dict)
            for nt in x_dict.keys():
                if i == 0:
                    x_dict[nt] = out_dict[nt]
                else:
                    x_dict[nt] = out_dict[nt] + x_dict[nt]
                x_dict[nt] = self.act(x_dict[nt])
                x_dict[nt] = self.dropout(x_dict[nt])
        z_dict = x_dict
        out = self.decoder(z_dict, edge_label_index)
        return z_dict, out

TOP_K = 5
NUM_EPOCHS = 10
DB_NAME = 'e'

class LinkPredictionEvaluator:
    def __init__(self, model, criterion, optimizer=None, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        _, pred = self.model(data.x_dict, data.edge_index_dict, data['drug', 'protein'].edge_label_index)
        target = data['drug', 'protein'].edge_label.float().to(self.device)
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        emb, pred = self.model(data.x_dict, data.edge_index_dict, data['drug', 'protein'].edge_label_index)
        target = data['drug', 'protein'].edge_label.float().to(self.device)
        loss = self.criterion(pred, target).item()
        prob = pred.view(-1).sigmoid().detach().cpu().numpy()
        label = target.detach().cpu().numpy()
        auc = roc_auc_score(label, prob)
        aupr = average_precision_score(label, prob)
        pred_label = (prob > 0.5).astype(int)
        acc = accuracy_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        return {'AUC': auc, 'AUPR': aupr, 'Loss': loss, 'ACC': acc, 'F1': f1, 'emb': emb, 'prob': prob}

def extract_darts_architecture_flexible(model):

    arch = []
    for layer in model.layers:
        layer_cfg = {}
        if hasattr(layer, 'alpha'):
            alpha = layer.alpha
            op_index = torch.argmax(F.softmax(alpha, dim=-1)).item()
            op_name = list(layer.ops.keys())[op_index]
            conv_type = op_name.lower()
            norm_type = layer.norm_choices[op_index] if hasattr(layer, 'norm_choices') else 'identity'
            act_type = layer.act_choices[op_index] if hasattr(layer, 'act_choices') else 'identity'
        else:
            conv_type = getattr(layer, 'conv_type', 'sage')
            norm_type = getattr(layer, 'norm_type', 'identity')
            act_type = getattr(layer, 'act_type', 'identity')

        layer_cfg['conv'] = conv_type
        layer_cfg['norm'] = norm_type
        layer_cfg['act'] = act_type
        arch.append(layer_cfg)
    return arch

def save_top_k_architectures_flexible(db_name, top_k_models):
    try:
        all_results_formatted = []
        for i, item in enumerate(top_k_models):
            model_state = item['model_state']
            reward = item['reward']
            model = item['model_instance']
            model.load_state_dict(model_state)
            arch = extract_darts_architecture_flexible(model)
            formatted_res = {
                'rank': i + 1,
                'combined_reward': reward,
                'architecture': arch
            }
            all_results_formatted.append(formatted_res)

        script_directory = 'Code/search_architecture/'
        results_dir = os.path.join(script_directory, 'results')
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, f"{db_name}_darts.json")
        with open(filepath, 'w') as f:
            json.dump(all_results_formatted, f, indent=4)
        print(f"\nSuccess! Saved the top {len(all_results_formatted)} best architectures to {filepath}")
    except Exception as e:
        print(f"\n[ERROR] Exception occurred while saving Top-K architectures: {e}")

def main():
    data_path = f"Data/{DB_NAME}/hetero_data_{DB_NAME}.pt"
    print(f"Loading data from: {data_path}")
    data = torch.load(data_path)
    data = T.ToUndirected()(data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    split = T.RandomLinkSplit(
        num_val=0.1, num_test=0.2, is_undirected=True,
        add_negative_train_samples=True, neg_sampling_ratio=1.0,
        disjoint_train_ratio=0.3,
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
    )
    train_data, val_data, _ = split(data)
    train_data, val_data = train_data.to(device), val_data.to(device)

    model = DARTSHeteroModel(data.metadata(), hidden_dim=60, num_layers=5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    evaluator = LinkPredictionEvaluator(model, criterion, optimizer, device)

    top_k_models = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = evaluator.train_one_epoch(train_data)
        val_metrics = evaluator.evaluate(val_data)
        reward = 0.5 * val_metrics['AUC'] + 0.5 * val_metrics['AUPR']

        top_k_models.append({
            'reward': reward,
            'model_state': copy.deepcopy(model.state_dict()),
            'model_instance': copy.deepcopy(model)
        })
        top_k_models = sorted(top_k_models, key=lambda x: x['reward'], reverse=True)[:TOP_K]

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_metrics['Loss']:.4f}, "
              f"val_auc={val_metrics['AUC']:.4f}, val_aupr={val_metrics['AUPR']:.4f}, reward={reward:.4f}")

    save_top_k_architectures_flexible(DB_NAME, top_k_models)

if __name__ == "__main__":
    main()