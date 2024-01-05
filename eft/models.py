import lightning.pytorch as pl
# from enformer_pytorch import Enformer
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch import seq_indices_to_one_hot
from eft.data import CustomGenomeIntervalDataset
import torch


class EnformerTX(pl.LightningModule):
    def __init__(self, pretrained_state_dict=None, learning_rate=3e-4):
        super().__init__()
        self.model = from_pretrained('EleutherAI/enformer-official-rough')
        self.model = HeadAdapterWrapper(
            enformer=self.model,
            num_tracks=1,
            post_transformer_embed=False
        )
        if pretrained_state_dict is not None:
            self.model.load_state_dict(pretrained_state_dict, strict=False)
        self.save_hyperparameters()

    def forward(self, sequence, target):
        # if self.device.type != 'cpu':
        #     sequence = sequence.to(dtype=torch.long)
        #     target = target.to(dtype=torch.float32)
        #     sequence = seq_indices_to_one_hot(sequence)
        return self.model(sequence, target=target)

    def training_step(self, batch, batch_idx):
        seq, target = batch
        loss = self(seq, target)
        self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        dataset = CustomGenomeIntervalDataset(
            bed_file='data/sequences/train_90_10.bed',
            fasta_file='data/hg38.fa',
            context_length=196_608
        )

        # fetch a random seq and target from the training dataset
        seq, target = dataset[0]
        seq = seq.to(self.device)

        # get the model's prediction
        pred = self(seq, None, return_embeddings=True)
        print(pred)
        print(pred.shape)

        # TODO: plot the results

        return 0

    def validation_step(self, batch, batch_idx):
        seq, target = batch
        loss = self(seq, target)
        self.log('val/loss', loss, sync_dist=True, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        seq, target = batch
        loss = self(seq, target)
        self.log('test/loss', loss, sync_dist=True, on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seq, _ = batch
        return self(seq, None)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
