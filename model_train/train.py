import torch
import os
import numpy as np
import torch.utils
import torch.utils.data
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
            class_dist = torch.zeros(( len(list_of_dirs)), dtype=torch.float32)
            class_dist[i] = 1.
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

        self.flatten = torch.nn.Flatten(start_dim= 1)
        
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
        #x = self.softmax(x)
        return x
    
    def fit(self,
            train_dataloader: torch.utils.data.DataLoader,
            validation_dataloader: torch.utils.data.DataLoader,
            learning_rate: float = 0.001,
            epochs: int = 5) -> None:
        
        # define model optimizer and loss function
        OPTIMIZER = torch.optim.Adam(self.parameters(), learning_rate)
        LOSS_FN = torch.nn.CrossEntropyLoss()

        for e in range(epochs):
            print("EPOCH: ", e+1)
            save_loss = 0.
            for i, data in enumerate(train_dataloader):
                inputs, labels = data

                # zerp gradients
                OPTIMIZER.zero_grad()

                # Forward pass:
                predictions = self.forward(inputs)

                # calculate loss and gradients
                loss = LOSS_FN(predictions, labels)
                loss.backward()

                # perform optimization
                OPTIMIZER.step()
                
                if i % 50 == 49:
                    print(f"    BATCH: {i+1} LOSS: {loss.item()}")
                    save_loss += loss.item()
            
            save_loss /= i+1

            # evaluate model every epoch
            with torch.no_grad():
                vloss = 0.
                for i, data in enumerate(validation_dataloader):
                    vinputs, vlabels = data
                    vpredicts = self.forward(vinputs)
                    vloss += LOSS_FN(predictions, labels)
                vloss /= i+1
                print(f"  Validation loss: {vloss}, loss: {save_loss}")






if __name__ == "__main__":
    dataset = MathCharactersDataset("dataset")
    TRAIN_TEST_SPLIT = 0.8

    # Split datasets into train and validation sets
    train_size = int(TRAIN_TEST_SPLIT * len(dataset)) # train size = 8056
    validation_size = len(dataset) - train_size # validation_size = 2015
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)

    model = CharModel()
    model.fit(train_dataloader,validation_dataloader, epochs=5)
        





