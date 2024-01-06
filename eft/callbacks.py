import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path

class CustomPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            print(f"Created output directory: {self.output_dir}")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, self.output_dir.joinpath(f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, self.output_dir.joinpath(f"batch_indices_{trainer.global_rank}.pt"))