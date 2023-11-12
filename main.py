from lightning.pytorch.cli import LightningCLI
from eft.models import EnformerTX
from eft.data import EnformerTXDataModule

def cli_main():
    cli = LightningCLI(EnformerTX, EnformerTXDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
