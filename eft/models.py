import lightning.pytorch as pl
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
import torch
from eft.viz import save_plot_to_pdf
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
from eft.data import convert_int_to_chr
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

class EnformerTX(pl.LightningModule):
    def __init__(self, pretrained_state_dict=None, learning_rate=3e-4, val_viz_interval=None):
        super().__init__()
        self.model = from_pretrained('EleutherAI/enformer-official-rough')
        self.model = HeadAdapterWrapper(
            enformer=self.model,
            num_tracks=1,
            post_transformer_embed=False
        )
        if pretrained_state_dict is not None:
            self.model.load_state_dict(pretrained_state_dict, strict=False)
        self.val_viz_interval = 1 if val_viz_interval == 0 else val_viz_interval
        self.validation_step_preds = []
        self.validation_step_targets = []
        self.validation_step_locs = []
        self.save_hyperparameters()

    def forward(self, sequence, target):
        return self.model(sequence, target=target)

    def training_step(self, batch, batch_idx):
        seq, target, loc = batch
        loss = self(seq, target)
        loss = torch.nan_to_num(loss)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, target, loc = batch
        loss = self(seq, target)
        loss = torch.nan_to_num(loss)
        self.log('val/loss', loss, sync_dist=True, on_step=True, on_epoch=True)
        if self.current_epoch % self.val_viz_interval == 0:
            n_samples = len(self.validation_step_preds) * len(seq) # Number of samples in each batch
            if n_samples < 100 and loss != 0:
                pred = self(seq, None)
                self.validation_step_preds.append(pred)
                self.validation_step_targets.append(target)
                self.validation_step_locs.append(loc)

    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch % self.val_viz_interval == 0) and (len(self.validation_step_preds) > 0):
            self.validation_step_preds = torch.concat(self.validation_step_preds).squeeze()
            self.validation_step_targets = torch.concat(self.validation_step_targets).squeeze()
            self.validation_step_locs = torch.concat(self.validation_step_locs)
            self.validation_step_preds = self.validation_step_preds.cpu().numpy()
            self.validation_step_targets = self.validation_step_targets.cpu().numpy()
            self.validation_step_locs = self.validation_step_locs.cpu().numpy()

            pdf_below_path = Path(self.logger.log_dir).joinpath(f'epoch={self.current_epoch}_rho-less-0.3.pdf')
            pdf_above_path = Path(self.logger.log_dir).joinpath(f'epoch={self.current_epoch}_rho-gre-0.3.pdf')

            with PdfPages(pdf_below_path) as pdf_below, PdfPages(pdf_above_path) as pdf_above:
                for i in range(len(self.validation_step_preds)):
                    true = self.validation_step_targets[i]
                    pred = self.validation_step_preds[i]
                    loc = tuple(self.validation_step_locs[i])
                    loc = (convert_int_to_chr(loc[0]),) + loc[1:]

                    rho, pval = pearsonr(pred, true)

                    if rho == float('nan'):
                        continue

                    if rho < 0.3:
                        save_plot_to_pdf(pdf_below, true, pred, loc, (rho, pval))
                    elif rho >= 0.3:
                        save_plot_to_pdf(pdf_above, true, pred, loc, (rho, pval))
        # reset lists
        self.validation_step_preds = []
        self.validation_step_targets = []
        self.validation_step_locs = []

    def test_step(self, batch, batch_idx):
        seq, target, loc = batch
        loss = self(seq, target)
        loss = torch.nan_to_num(loss)
        self.log('test/loss', loss, sync_dist=True, on_step=True, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seq, target, loc = batch
        return self(seq, None)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # Get max_epochs from the trainer
        if self.trainer:
            max_epochs = self.trainer.max_epochs
        else:
            max_epochs = 50  # Default value if the trainer is not yet attached

        # Configure warmup and annealing
        warmup_epochs = 5
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min(1., epoch / warmup_epochs))
        anneal_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=0)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": SequentialLR(optimizer, schedulers=[warmup_scheduler, anneal_scheduler], milestones=[warmup_epochs]),
                "interval": "epoch",
                "frequency": 1,
            },
        }
