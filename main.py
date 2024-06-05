from model.main_model import MainModel, MainGraph
from dataset.GraphDataset import GraphDataModule
import pytorch_lightning as pl

if __name__ == '__main__':
    dm = GraphDataModule(batch_size=256, shuffle=True, drop_last=False,
                         root_path='/Users/mm/developer/PycharmProject/myRCA/data/pyg_dataset/TT-all',
                         raw_path='/Users/mm/developer/PycharmProject/myRCA/data/TT Dataset/all')
    model = MainModel()
    # 定义数据集为训练校验阶段
    dm.setup('fit')
    trainer = pl.Trainer(accelerator="cpu", max_epochs=150)
    trainer.fit(model, datamodule=dm)
