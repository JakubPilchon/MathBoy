import torch
import os
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Grayscale, RandomRotation, Compose, ConvertImageDtype
from typing import Tuple
## set device

class MathCharactersDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str | os.PathLike):
        assert os.path.isdir(dir)
        super().__init__()

        self.dir = dir
        self.data = []
        list_of_dirs = os.listdir(self.dir)

        self.transforms = Compose([
            Grayscale(num_output_channels=1),
            ConvertImageDtype(torch.float),
            RandomRotation((-30,30)),
            ])
        
        ## get dataset annotations
        for i, class_name in enumerate(list_of_dirs):
            class_dist = torch.zeros((1, len(list_of_dirs)), dtype=torch.float32)
            class_dist[0][i] = 1.
            for data_name in os.listdir(os.path.join(self.dir, class_name)):
                self.data.append((os.path.join(self.dir, class_name, data_name), class_dist))

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, str]:
        if not isinstance(index, slice):
            img, class_name = self.data[index]
            img = read_image(img)
            img = self.transforms(img)
            return (img, class_name)
        else:
            raise TypeError("MathCharactersDataset does not support slicing")


class CharModel(torch.nn.Module):
    def __init__(self):
        super(CharModel, self).__init__()

        # define model architecture
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.padd1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(32, 16, 3, 1)
        self.padd2 = torch.nn.MaxPool2d(3,3)

        self.flatten = torch.nn.Flatten(start_dim=0)
        
        self.linear1 = torch.nn.Linear(256, 64)
        self.linear2 = torch.nn.Linear(64, 19)

        self.activation = torch.nn.functional.relu
        self.softmax = torch.nn.Softmax()
        

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.padd1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.padd2(x)
        x = self.activation(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.softmax(x)
        return x
        




if __name__ == "__main__":
    dataset = MathCharactersDataset("dataset")
    img,  target = dataset[5000]
    model = CharModel()
    predict = model.forward(img)
    print()
        
        





