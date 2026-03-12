import os
import time

import torch_geometric.transforms as T
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from Code.search_architecture.lstm_search.auto_model import FlexibleGNNModel


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
        return {
            'AUC': auc,
            'AUPR': aupr,
            'Loss': loss,
            'ACC': acc,
            'F1': f1,
            'emb': emb,
            'prob': prob
        }


if __name__ == "__main__":
    DATABASE = 'nr'
    PATH_DATA = os.path.join('Data', DATABASE.upper(), f'hetero_data_{DATABASE}.pt')
    device = 'cuda'
    print(device)
    data = torch.load(PATH_DATA).to(device)
    print(f'Data is cuda?: {data.is_cuda}')
    data = T.ToUndirected()(data)
    del data['protein', 'rev_interaction', 'drug'].edge_label

    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=True,  # False for: Not adding negative links to train
        neg_sampling_ratio=1.0,  # ratio of negative sampling is 0
        disjoint_train_ratio=0.3,
        edge_types=[('drug', 'interaction', 'protein')],
        rev_edge_types=[('protein', 'rev_interaction', 'drug')],
        split_labels=False
    )

    train_data, val_data, test_data = split(data)


    layer_configs = [
        {'conv': 'transformerconv', 'norm': 'instancenorm', 'act': 'tanh'},
        {'conv': 'sage', 'norm': 'graphnorm', 'act': 'relu'},
        {'conv': 'gatconv', 'norm': 'layernorm', 'act': 'tanh'},
        {'conv': 'transformerconv', 'norm': 'instancenorm', 'act': 'relu'},
        {'conv': 'gatconv', 'norm': 'layernorm', 'act': 'tanh'}
    ]

    model = FlexibleGNNModel(layer_configs, data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    evaluator = LinkPredictionEvaluator(model, criterion, optimizer, device)

    init_time = time.time()
    for epoch in range(500):
        train_loss = evaluator.train_one_epoch(train_data)

        if epoch % 50 == 0:
            val_metrics = evaluator.evaluate(val_data)
            # test_metrics = evaluator.evaluate(test_data)

            print(f"Epoch {epoch:03d} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f} ACC: {val_metrics['ACC']:.4f} F1: {val_metrics['F1']:.4f} |")

    end_time = time.time()
    print(f"used time {(end_time - init_time) / 60:.4f} min")