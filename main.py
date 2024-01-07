import torch
from lightning.pytorch.cli import LightningCLI
from eft.models import EnformerTX
from eft.data import EnformerTXDataModule
torch.set_float32_matmul_precision('high')


def cli_main():
    cli = LightningCLI(EnformerTX, EnformerTXDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory_gb = total_memory / (1024 ** 3)  # Convert bytes to GB
        print(f"Device: {device_name}, Total VRAM: {total_memory_gb:.2f} GB")
    cli_main()

