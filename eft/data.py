import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import random_split, DataLoader, ConcatDataset
from enformer_pytorch import GenomeIntervalDataset
from pathlib import Path
import torch
import ast
import eft
from sklearn.model_selection import train_test_split


class CustomGenomeIntervalDataset(GenomeIntervalDataset):
    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, seq_type, target = (interval[0], interval[1], interval[2], interval[3], interval[4])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        target = ast.literal_eval(target)
        target = torch.tensor(target)

        sequence = self.fasta(chr_name, start, end, return_augs=self.return_augs)

        return sequence, target


class EnformerTXDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = None, batch_size: int = 32, num_workers: int = 4, dev = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dev = dev
        self.save_hyperparameters()
        self.fasta_file = str(Path(eft.__file__).parents[1].joinpath('data/hg38.fa'))

    def prepare_data(self):
        train_path = self.data_dir.joinpath('train.bed')
        val_path = self.data_dir.joinpath('val.bed')
        if not train_path.exists() and not val_path.exists():
            self.split_data()

    def split_data(self):
        df = pd.read_csv(self.data_dir.joinpath('promoter_dnase.bed'), sep='\t', header=None)
        df.columns = ['chr', 'start', 'end', 'seq_type', 'target']
        train, val = train_test_split(df, test_size=0.2, random_state=42)
        train.to_csv(self.data_dir.joinpath('train.bed'), sep='\t', header=False, index=False)
        val.to_csv(self.data_dir.joinpath('val.bed'), sep='\t', header=False, index=False)
        train.sample(n=500).to_csv(self.data_dir.joinpath('train_dev.bed'), sep='\t', header=False, index=False)
        val.sample(n=50).to_csv(self.data_dir.joinpath('val_dev.bed'), sep='\t', header=False, index=False)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_file = 'train_dev.bed' if self.dev else 'train_90_10.bed'
            val_file = 'val_dev.bed' if self.dev else 'val_90_10.bed'
            self.train = CustomGenomeIntervalDataset(
                bed_file=self.data_dir / train_file,
                fasta_file=self.fasta_file,
                return_seq_indices=True,
                context_length=196_608,
            )
            self.val = CustomGenomeIntervalDataset(
                bed_file=self.data_dir / val_file,
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
