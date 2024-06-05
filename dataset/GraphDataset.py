import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from utils.graph_utils import prepare
import torch.utils.data as data
import torch


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, shuffle, drop_last, root_path, raw_path):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.root_path = root_path
        self.raw_path = raw_path

    def setup(self, stage):
        ds = RCADataset(root=self.root_path, data_path=self.raw_path)
        train_set_size = int(len(ds) * 0.6)
        valid_set_size = int(len(ds) * 0.2)
        test_set_size = int(len(ds) * 0.2 + 2)
        self.train_dataset, self.val_dataset, self.test_dataset = data.random_split(ds, [train_set_size, valid_set_size,
                                                                                         test_set_size])
        # self.train_dataset, self.val_dataset, self.test_dataset = ds, ds, ds

    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)


class RCADataset(InMemoryDataset):
    def __init__(self, root, data_path, transform=None, pre_transform=None, pre_filter=None):
        self.data_path = data_path
        super(RCADataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = prepare(self.data_path)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
