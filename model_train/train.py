import torch
import os
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Grayscale, RandomRotation, Compose
from typing import Tuple
## set device

class MathCharactersDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str | os.PathLike):
        assert os.path.isdir(dir)
        super().__init__()

        self.dir = dir
        self.data = []
        self.transforms = Compose([
            Grayscale(num_output_channels=1),
            RandomRotation((-30,30))
            ])
        
        ## get dataset annotations
        for class_name in os.listdir(self.dir):
            for data_name in os.listdir(os.path.join(self.dir, class_name)):
                self.data.append((os.path.join(self.dir, class_name, data_name), class_name))


    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, str]:
        if not isinstance(index, slice):
            img, class_name = self.data[index]
            img = read_image(img)
            img = self.transforms(img)
            return (img, class_name)
        else:
            raise TypeError("athCharactersDataset does not support slicing")






if __name__ == "__main__":
    dataset = MathCharactersDataset("dataset")
    #print(dataset.data[])
    print(len(dataset))
    print(dataset[3])

        
        





