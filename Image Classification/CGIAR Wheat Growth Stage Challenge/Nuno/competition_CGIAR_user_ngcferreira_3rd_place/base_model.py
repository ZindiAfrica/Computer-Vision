import os
import torch
from pathlib import Path
from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def checkpoint_name(self, prefix=None) -> str:
        checkpoint = f'{self.model_name}_{self.dropout}_{self.neurons}.pth'
        checkpoint = f'{prefix}_{checkpoint}' if prefix else checkpoint
        return checkpoint

    def __get_saved_model_filename(self, directory=None, filename=None, prefix=None) -> str:
        filename = filename if filename else self.checkpoint_name(prefix=prefix)
        filename = os.path.join(directory, filename) if directory else filename
        return filename

    def save(self, directory=None, filename=None, prefix=None) -> None:
        filename = self.__get_saved_model_filename(directory=directory, filename=filename, prefix=prefix)
        Path(directory).mkdir(parents=True, exist_ok=True)

        print(f"Saving weights to {filename}")
        torch.save(self.state_dict(), filename)

    def load(self, directory=None, filename=None, prefix=None) -> None:
        filename = self.__get_saved_model_filename(directory=directory, filename=filename, prefix=prefix)
        if os.path.exists(filename):
            print(f"Loading saved weights from {filename}")
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)
        else:
            print("======================================================")
            print(f"WARNING: No weights available in {filename}")
            print("======================================================")
