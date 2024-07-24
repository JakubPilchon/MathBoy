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

        # MODEL ARCHITECTURE
        # first convolutional block
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1)
        self.padd1 = torch.nn.MaxPool2d(2, 2)
        self.bn1 = torch.nn.BatchNorm2d(16)

        # second convolutional block
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1)
        self.padd2 = torch.nn.MaxPool2d(2, 2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        

        # third convolutional block
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1)
        self.padd3 = torch.nn.MaxPool2d(2, 2)
        self.bn3 = torch.nn.BatchNorm2d(64)

        # Linear block
        #    flatten convolutions into vectors
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(256, 64)
        self.linear2 = torch.nn.Linear(64, 19)

        #    Activation added inbetween blocks
        self.activation = torch.nn.functional.relu

        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropoutcnn = torch.nn.Dropout2d(0.3)
        

    def forward(self,
                x: torch.tensor,
                is_training: bool = False
                ) -> torch.tensor:

        # Flow of single character image
        # input shape = (1, 1,32,32)
        if len(x.shape) == 3: #change shape if input is 3D tensor
            x = x.reshape((1,1,32,32))

        x = self.conv1(x) # shape = (1,16,30,30)
        x = self.padd1(x) # shape = (1,16,15,15)
        x = self.bn1(x) # shape = (1,16,15,15)
        x = self.activation(x) # shape = (1,16,15,15)
        if is_training:
            x = self.dropoutcnn(x)

        x = self.conv2(x) # shape = (1,32,13,13)
        x = self.padd2(x) # shape = (1,32,6,6)
        x = self.bn2(x)
        x = self.activation(x) # shape = (1,32,6,6)
        if is_training:
            x = self.dropoutcnn(x)

        x = self.conv3(x) # shape = (1,64,4,4)
        x = self.padd3(x) # shape = (1,64,2,2)
        x = self.bn3(x) # shape = (1,64,2,2)
        x = self.activation(x) # shape = (1,64,2,2)  
        if is_training:
            x = self.dropoutcnn(x)

        x = self.flatten(x) # shape (1,256)

        x = self.linear1(x) # shape = (1,64)

        if is_training:
            x = self.dropout1(x)

        x = self.activation(x) # shape = (1,64)
        x = self.linear2(x) # shape = (1,19)
        return x
    
    def accuracy(self,
                 predicts: torch.tensor,
                 labels: torch.tensor
                 ) -> torch.float:
        predicts = torch.argmax(predicts, dim=1)
        labels = torch.argmax(labels, dim=1)
        return torch.sum(predicts.eq(labels).float())
    
    def fit(self,
            train_dataloader: torch.utils.data.DataLoader,
            validation_dataloader: torch.utils.data.DataLoader,
            learning_rate: float = 0.001,
            epochs: int = 5
            ) -> None:
        
        # define model optimizer and loss function
        OPTIMIZER = torch.optim.Adam(self.parameters(), learning_rate)
        LOSS_FN = torch.nn.CrossEntropyLoss()

        # threshold for best validation loss
        best_vloss = float("inf")

        for e in range(epochs):
            print("EPOCH: ", e+1)
            save_loss = 0.
            for i, data in enumerate(train_dataloader):
                inputs, labels = data

                # zerp gradients
                OPTIMIZER.zero_grad()

                # Forward pass:
                predictions = self.forward(inputs, is_training= True)

                # calculate loss and gradients
                loss = LOSS_FN(predictions, labels)
                loss.backward()

                # perform optimization
                OPTIMIZER.step()
                
                if i % 5 == 4:
                    print(u"    Batch: {} \u2551{}{}\u2551 Loss: {}".format(
                        str(i+1).zfill(3), # batch number
                        u"\u2588" * int((i+1)/5), # progress bar which is completed
                        " " * int(len(train_dataloader)/5 - (i+1)/5), # progress bar which is empty
                        loss.item()), end="\r") # loss value, end arguments makes print() function replace last displayed message in console
                    save_loss += loss.item()
            
            #print("\n")
            save_loss /= i

            # evaluate model every epoch, disabling gradients
            with torch.no_grad():
                vloss = 0.
                vacc = 0.
                for i, data in enumerate(validation_dataloader):
                    vinputs, vlabels = data
                    vpredicts = self.forward(vinputs)
                    vloss += LOSS_FN(vpredicts, vlabels)
                    vacc += self.accuracy(vpredicts, vlabels)
                vloss /= i+1
                vacc /=  (len(validation_dataloader) * validation_dataloader.batch_size)
                print(f"  Validation loss: {vloss},  validation accuracy: {vacc}, loss: {save_loss}")

                if vloss < best_vloss:
                    best_vloss = vloss
                    torch.save(self.state_dict(), "model.pt")
                    print('\33[32m' + "  Best performance up to date, model saved.", '\33[0m')
                else:
                    print('\33[31m', "  Smaller validation loss, model wasn't saved", '\33[0m')
            
                
if __name__ == "__main__":
    dataset = MathCharactersDataset("dataset")

    TRAIN_TEST_SPLIT = 0.8
    LEARNING_RATE = .01

    # Split datasets into train and validation sets
    train_size = int(TRAIN_TEST_SPLIT * len(dataset)) # train size = 8056
    validation_size = len(dataset) - train_size # validation_size = 2015
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=True)

    model = CharModel()
    model.fit(train_dataloader,
            validation_dataloader,
            learning_rate=LEARNING_RATE,
            epochs=25)
        