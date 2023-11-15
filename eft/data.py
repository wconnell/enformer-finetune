import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import random_split, DataLoader, ConcatDataset
from enformer_pytorch import GenomeIntervalDataset
from pathlib import Path
import torch
import ast
import eft


class CustomGenomeIntervalDataset(GenomeIntervalDataset):
    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, seq_type, target = (interval[0], interval[1], interval[2], interval[3], interval[4])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        target = ast.literal_eval(target)
        target = torch.tensor(target)
        return self.fasta(chr_name, start, end, return_augs=self.return_augs), target


class EnformerTXDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = None, batch_size: int = 32, num_workers: int = 4, train_test_split: float = 0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_test_split = train_test_split
        self.save_hyperparameters()
        self.fasta_file = str(Path(eft.__file__).parents[1].joinpath('data/hg38.fa'))

    def prepare_data(self):
        full_data = pd.read_csv(self.data_dir.joinpath('promoter_dnase.bed'), sep='\t', header=None)
        full_data.columns = ['chr', 'start', 'end', 'seq_type', 'target']

        promoter_data = full_data[full_data['seq_type'] == 'promoter']
        control_data = full_data[full_data['seq_type'] == 'random']

        train_size = int(len(promoter_data) * self.train_test_split)

        promoter_train = promoter_data[:train_size]
        promoter_val = promoter_data[train_size:]
        control_train = control_data[:train_size]
        control_val = control_data[train_size:]

        train = pd.concat([promoter_train, control_train])
        val = pd.concat([promoter_val, control_val])
        train.to_csv(self.data_dir.joinpath('train.bed'), sep='\t', header=False, index=False)
        val.to_csv(self.data_dir.joinpath('val.bed'), sep='\t', header=False, index=False)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = CustomGenomeIntervalDataset(
                bed_file=self.data_dir.joinpath('train.bed'),
                fasta_file=self.fasta_file,
                return_seq_indices=True,
                context_length=196_608,
            )
            self.val = CustomGenomeIntervalDataset(
                bed_file=self.data_dir.joinpath('val.bed'),
                fasta_file=self.fasta_file,
                return_seq_indices=True,
                context_length=196_608,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            raise NotImplementedError("Test data not implemented")

        if stage == "predict":
            raise NotImplementedError("Predict data not implemented")

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return val_loader

    def test_dataloader(self):
        raise NotImplementedError("Test data not implemented")

    def predict_dataloader(self):
        raise NotImplementedError("Predict data not implemented")
