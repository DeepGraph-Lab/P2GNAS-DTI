
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.nn import (
    SAGEConv, GraphConv, TransformerConv, GATConv, GATv2Conv,
    GraphNorm, InstanceNorm, LayerNorm, BatchNorm, to_hetero
)

CONV_LAYERS = {
    'sage': SAGEConv,
    'graphconv': GraphConv,
    'transformerconv': TransformerConv,
    'gatconv': GATConv,
    'gatv2conv': GATv2Conv,
}
NORM_LAYERS = {
    'graphnorm': GraphNorm,
    'instancenorm': InstanceNorm,
    'layernorm': LayerNorm,
    'batchnorm': BatchNorm,
}

ACT_LAYERS = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leakyrelu': nn.LeakyReLU(),
    'identity': nn.Identity(),
}


class FlexibleGNNEncoder(nn.Module):
    def __init__(self, layer_configs=None):
        super().__init__()

        hidden_dim = 60  #in FlexibleGNNModel also have hidden_channels should change
        aggr = 'sum'
        dropout = 0.2
        heads = 2  ##make sure: hidden_din % heads = 0

        if layer_configs is None:
            layer_configs = [
                {'conv': 'sage', 'norm': 'graphnorm', 'act': 'relu'},
                {'conv': 'gatconv', 'norm': 'layernorm', 'act': 'tanh'},
                {'conv': 'transformerconv', 'norm': 'instancenorm', 'act': 'relu'},
            ]

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i, cfg in enumerate(layer_configs):
            conv_type = cfg['conv'].lower()
            in_dim = hidden_dim if i == 0 else hidden_dim

            if conv_type in ['gatconv', 'gatv2conv']:
                conv = CONV_LAYERS[conv_type](
                    (-1, -1), hidden_dim// heads, heads=heads, add_self_loops=False
                )
            elif conv_type == 'transformerconv':
                conv = TransformerConv((-1, -1),hidden_dim// heads, heads=heads)
            else:
                conv = CONV_LAYERS[conv_type](
                    (-1, -1), hidden_dim, aggr=aggr
                )
            self.layers.append(conv)
            # Norm
            norm_type = cfg.get('norm', 'graphnorm').lower()
            norm = NORM_LAYERS[norm_type](hidden_dim) if norm_type in NORM_LAYERS else nn.Identity()
            self.norms.append(norm)

            act_type = cfg.get('act', 'relu').lower()
            act = ACT_LAYERS.get(act_type, nn.Identity())
            self.acts.append(act)
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x, edge_index):

        if not self.layers:
            return x


        x = self.layers[0](x, edge_index)
        x = self.norms[0](x)
        x = self.acts[0](x)
        x = self.dropouts[0](x)


        for i in range(1, len(self.layers)):
            x_residual = x

            x = self.layers[i](x, edge_index)
            x = self.norms[i](x)

            x = x + x_residual

            x = self.acts[i](x)
            x = self.dropouts[i](x)

        return x

class EdgeClassifier(torch.nn.Module):
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


class FlexibleGNNModel(torch.nn.Module):
    def __init__(self, layer_configs, data_metadata=None):

        super().__init__()
        self.encoder = FlexibleGNNEncoder(layer_configs)
        if data_metadata is not None:
            self.encoder = to_hetero(self.encoder, data_metadata, aggr='sum')
        hidden_channels = 60
        self.decoder = EdgeClassifier(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out = self.decoder(z_dict, edge_label_index)
        return z_dict, out


def main():
    layer_configs = [
        {'conv': 'transformerconv', 'norm': 'instancenorm', 'act': 'tanh'},
        {'conv': 'sage', 'norm': 'graphnorm', 'act': 'relu'},
        {'conv': 'gatconv', 'norm': 'layernorm', 'act': 'tanh'},
        {'conv': 'transformerconv', 'norm': 'instancenorm', 'act': 'relu'},
        {'conv': 'gatconv', 'norm': 'layernorm', 'act': 'tanh'}
    ]

    num_drug = 6
    num_protein = 8
    feat_dim = 32
    drug_x = torch.randn(num_drug, feat_dim)
    protein_x = torch.randn(num_protein, feat_dim)

    edge_index = torch.tensor([
        [0, 1, 2, 2, 3, 4, 4, 5, 0, 3],  # drug
        [1, 3, 0, 2, 4, 6, 7, 5, 2, 7]  # protein
    ], dtype=torch.long)  # shape: [2, num_edges]

    data = HeteroData()
    data['drug'].x = drug_x
    data['protein'].x = protein_x
    data['drug', 'interaction', 'protein'].edge_index = edge_index

    # print(data)
    # print("drug.x.shape:", data['drug'].x.shape)
    # print("protein.x.shape:", data['protein'].x.shape)
    # print("edge_index.shape:", data['drug', 'interaction', 'protein'].edge_index.shape)
    device = 'cuda'

    data = T.ToUndirected()(data)
    # Remove "reverse" label.
    del data['protein', 'rev_interaction', 'drug'].edge_label
    data = data.to(device)

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
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = FlexibleGNNModel(layer_configs, data.metadata()).to(device)
    z_dict, pred = model(train_data.x_dict, train_data.edge_index_dict,
                    train_data['drug', 'protein'].edge_label_index)


    print("z_dict keys:", z_dict.keys())
    print("out.shape:", pred .shape)
    print("out:",pred)
    print("============all model structure============")
    print(model)
    print("============encoder model structure============")
    print(model.encoder)


if __name__ == "__main__":
    main()