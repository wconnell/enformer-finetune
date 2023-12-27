import lightning.pytorch as pl
# from enformer_pytorch import Enformer
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch import seq_indices_to_one_hot
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
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    # def on_train_epoch_end(self):
    #     outputs = self.trainer.fit_loop.epoch_loop.batch_loop.outputs
    #     avg_loss = torch.stack([x for x in outputs]).mean()
    #     self.log('train_loss_epoch', avg_loss)

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
