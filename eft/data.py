import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import random_split, DataLoader
from enformer_pytorch import GenomeIntervalDataset
from pathlib import Path
import torch
import ast
import eft

class CustomGenomeIntervalDataset(GenomeIntervalDataset):
    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, target = (interval[0], interval[1], interval[2], interval[3])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        target = ast.literal_eval(target)
        target = torch.tensor(target)
        return self.fasta(chr_name, start, end, return_augs = self.return_augs), target

class EnformerTXDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = None, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()
        self.fasta_file = str(Path(eft.__file__).parents[1].joinpath('data/hg38.fa'))
        

    def prepare_data(self):
        # TODO: Implement this
        full_data = pd.read_csv('../sequences/promoter_dnase.bed', sep='\t', header=None)
        full_data.columns = ['chr', 'start', 'end', 'target']
        return 0


    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = CustomGenomeIntervalDataset(
                bed_file = self.data_dir.joinpath('train.bed'),
                fasta_file = self.fasta_file,
                return_seq_indices = True,
                context_length = 196_608,
            )
            self.val = CustomGenomeIntervalDataset(
                bed_file = self.data_dir.joinpath('val.bed'),
                fasta_file = self.fasta_file,
                return_seq_indices = True,
                context_length = 196_608,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            raise NotImplementedError("Test data not implemented")

        if stage == "predict":
            raise NotImplementedError("Predict data not implemented")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        raise NotImplementedError("Test data not implemented")

    def predict_dataloader(self):
        raise NotImplementedError("Predict data not implemented")