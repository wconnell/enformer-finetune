import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import random_split, DataLoader, ConcatDataset
from enformer_pytorch import GenomeIntervalDataset
from pathlib import Path
import torch
import ast
import eft
from sklearn.model_selection import train_test_split
import gc

def convert_chr_to_int(x):
    if x == 'chrX':
        return 23
    elif x == 'chrY':
        return 24
    else:
        return int(x[3:])

def convert_int_to_chr(x):
    if x == 23:
        return 'chrX'
    elif x == 24:
        return 'chrY'
    else:
        return f'chr{x}'

class CustomGenomeIntervalDataset(GenomeIntervalDataset):
    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, ind):
        interval = self.df.row(ind)
        chr_name, start, end, seq_type, target = (interval[0], interval[1], interval[2], interval[3], interval[4])
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
        target = ast.literal_eval(target)
        target = torch.tensor(target).unsqueeze(-1)
        sequence = self.fasta(chr_name, start, end, return_augs=self.return_augs)
        chr_name = convert_chr_to_int(chr_name)
        loc = torch.tensor([chr_name, start, end])
        return sequence, target, loc

    @staticmethod
    def check_tensor_dtype(tensor):
        if tensor.dtype == torch.float16:
            return "fp16"
        elif tensor.dtype == torch.bfloat16:
            return "bf16"
        else:
            return str(tensor.dtype)


class EnformerTXDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path = None, batch_size: int = 32, num_workers: int = 4, dev = False, predict_data_path = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dev = dev
        self.predict_data_path = predict_data_path
        self.save_hyperparameters()
        self.fasta_file = str(Path(eft.__file__).parents[1].joinpath('data/hg38.fa'))

    @staticmethod
    def on_epoch_end():
        print('Clearing memory...')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            train_file = 'train_dev.bed' if self.dev else 'train.bed'
            val_file = 'val_dev.bed' if self.dev else 'val.bed'
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
            self.predict = CustomGenomeIntervalDataset(
                bed_file=self.predict_data_path,
                fasta_file=self.fasta_file,
                return_seq_indices=True,
                context_length=196_608,
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=None
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=None
        )
        return val_loader

    def test_dataloader(self):
        raise NotImplementedError("Test data not implemented")

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=None
        )
        return pred_loader

class MemoryLoggingCallback(pl.Callback):
    @staticmethod
    def on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx):
        memory_allocated = torch.cuda.memory_allocated()
        memory_cached = torch.cuda.memory_cached()
        trainer.logger.experiment.add_scalar("Memory/Allocated", memory_allocated, global_step=trainer.global_step)
        trainer.logger.experiment.add_scalar("Memory/Cached", memory_cached, global_step=trainer.global_step)
