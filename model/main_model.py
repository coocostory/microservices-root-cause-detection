import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GIN, GraphSAGE
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from .common import MLP
import torch.optim as optim
import pytorch_lightning as pl
from utils.metric_utils import cacl_metrics


class MainGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim, services_num, n_classes):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        # self.gin = GIN(in_dim, hidden_dim, 2)
        # self.gin = GraphSAGE(in_dim, hidden_dim, 2)
        # self.linear = nn.Linear(hidden_dim, 2)
        # self.linear = MLP(input_size=hidden_dim, hidden_sizes=hidden_sizes, output_size=output_size)
        self.conv1 = SAGEConv(in_channels=in_dim, out_channels=hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr='mean')
        self.classify1 = nn.Linear(hidden_dim, services_num)
        self.classify2 = nn.Linear(services_num, n_classes)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index, batch):
        h = x
        h = self.norm(h)
        h = F.elu(self.conv1(h, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        read_out = global_max_pool(h, batch)
        rv = self.classify1(read_out)
        h = self.classify2(rv)

        return h, rv


class BaseModel(pl.LightningModule):
    def __init__(self, model, learning_rate=0.01, weight_decay=1e-4, **kwargs):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_step_pred = []
        self.validation_step_label = []
        self.test_step_pred = []
        self.test_step_label = []
        self.loss_module = nn.CrossEntropyLoss()

    def validation_step(self, batch_g, i):
        h, _ = self.model(batch_g.x, batch_g.edge_index, batch_g.batch)
        loss = self.loss_module(h, batch_g.label)
        pred = torch.softmax(h, -1)
        pred = torch.max(pred, 1)[1].view(-1)
        self.validation_step_pred += pred.detach().cpu().numpy().tolist()
        self.validation_step_label += batch_g.label.cpu().numpy().tolist()
        return loss

    def on_validation_end(self):
        self.print("Val Metrics:", cacl_metrics(self.validation_step_label, self.validation_step_pred))
        pass

    def test_step(self, batch_g, i):
        h = self.model(batch_g.x, batch_g.edge_index, batch_g.batch)
        pred = torch.softmax(h, -1)
        pred = torch.max(pred, 1)[1].view(-1)
        self.test_step_pred += pred.detach().cpu().numpy().tolist()
        self.test_step_label += batch_g.label.cpu().numpy().tolist()

    def on_test_end(self, outputs):
        # self.print("Test Metrics:", cacl_metrics(self.test_step_label, self.test_step_pred))
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)


class MainModel(BaseModel):
    def __init__(self, in_dim=6, hidden_dim=16, hidden_sizes=[16, 32, 16], output_size=2):
        model = MainGraph(in_dim=6, services_num=27, n_classes=2, hidden_dim=64)
        # model = ClassifyGIN(6, 64)
        super().__init__(model=model)

    def training_step(self, batch_g):
        h, rv = self.model(batch_g.x, batch_g.edge_index, batch_g.batch)

        r_label = batch_g.r_label
        num_elements = r_label.size(0)
        num_cols = num_elements // batch_g.num_graphs
        num_rows = num_elements // num_cols
        r_label = r_label[:num_rows * num_cols].view(num_rows, num_cols)

        loss1 = self.loss_module(h, batch_g.label)
        loss2 = self.loss_module(rv, r_label)

        return loss1 + loss2
        # return loss1


class ClassifyGIN(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim):
        super(ClassifyGIN, self).__init__()
        self.save_hyperparameters()  # Save model hyperparameters for logging and checkpointing
        self.norm = nn.BatchNorm1d(in_dim)
        self.model = GIN(in_dim, hidden_dim, 2)
        self.linear = nn.Linear(hidden_dim, 2)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch):
        h = x
        h = self.norm(h)
        h = self.model(h, edge_index)
        read_out = global_max_pool(h, batch)
        h = self.linear(read_out)
        return h, 1

    def training_step(self, batch, batch_idx):
        x, edge_index, batch, label = batch.x, batch.edge_index, batch.batch, batch.label
        h = self(x, edge_index, batch)
        loss = self.loss_func(h, label)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0025, weight_decay=1e-4)
        return optimizer
