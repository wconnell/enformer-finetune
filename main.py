import torch
from lightning.pytorch.cli import LightningCLI
from eft.models import EnformerTX
from eft.data import EnformerTXDataModule

torch.set_float32_matmul_precision('medium')

def cli_main():
    cli = LightningCLI(EnformerTX, EnformerTXDataModule)
    # note: don't qcall fit!!


if __name__ == "__main__":
    cli_main()

    # TODO: We only have 40 GB VRAM. Figure out how many VRAM we need, and try to fix appropiately.
    #  - We can try to use a smaller batch size.
    #  - Parallelise: https://pytorch-lightning.readthedocs.io/en/1.1.8/multi_gpu.html#model-parallelism



