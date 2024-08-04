import torch
import os
import torch.utils
import torch.utils.data
from torchvision.io import read_image
from torchvision.transforms.v2 import Grayscale, RandomRotation, Compose, ConvertImageDtype, Lambda
from torchvision.transforms.v2.functional import invert
from typing import Tuple, Dict, List
## set device

class MathCharactersDataset(torch.utils.data.Dataset):
    def __init__(self, dir: str | os.PathLike):
        assert os.path.isdir(dir), "Provided directory is not valid"
        super().__init__()

        self.dir: str = dir
        self.data: List[Tuple[os.PathLike | str, str]] = []
        self.list_of_classes = []

        # labels pairs maping number to string representation
        self.labels: Dict[int, str] = {}
        # maping string to number representation
        self.rev_labels: Dict[str, int] = {}

        self.transforms = Compose([
            Grayscale(num_output_channels=1),
            Lambda(invert),
            ConvertImageDtype(torch.float),
            RandomRotation((-10,10))
            ])
        
        ## get dataset annotations
        self.num_classes = 0
        for file in os.listdir(dir):
            classname = file.split('-')[0]

            if classname not in self.rev_labels.keys():
                self.labels[self.num_classes] = classname
                self.rev_labels[classname] = self.num_classes
                self.num_classes += 1

            self.data.append((os.path.join(self.dir, file), self.rev_labels[classname]))

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        if not isinstance(index, slice):
            img, class_name = self.data[index]
            img = read_image(img)
            img = self.transforms(img)

            target = torch.zeros(self.num_classes)
            target[class_name] = 1.

            return (img, target)
        else:
            raise TypeError("MathCharactersDataset does not support slicing")


class CharModel(torch.nn.Module):
    def __init__(self):
        super(CharModel, self).__init__()

        # MODEL ARCHITECTURE
        # first convolutional block
        self.conv1 = torch.nn.Conv2d(1, 16, 3, 1, 1)
        self.padd1 = torch.nn.MaxPool2d(2, 2)
        self.bn1 = torch.nn.BatchNorm2d(16)

        # second convolutional block
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.padd2 = torch.nn.MaxPool2d(2, 2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        

        # third convolutional block
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.padd3 = torch.nn.MaxPool2d(2, 2)
        self.bn3 = torch.nn.BatchNorm2d(64)

        # Linear block
        #    flatten convolutions into vectors
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(1024, 128)
        self.linear2 = torch.nn.Linear(128,64)
        self.linear3 = torch.nn.Linear(64, 19)
        

        #    Activation added inbetween blocks
        self.activation = torch.nn.functional.relu

        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropoutcnn = torch.nn.Dropout2d(0.3)
        

    def forward(self,
                x: torch.tensor,
                is_training: bool = False
                ) -> torch.tensor:
        """Forward pass of convolutional neural network"""
        # Flow of single character image
        # input shape = (1, 1,32,32)
        if len(x.shape) == 3: #change shape if input is 3D tensor
            x = x.reshape((1,1,32,32))

        x = self.conv1(x) # shape = (1,16,30,30)
        x = self.padd1(x) # shape = (1,16,15,15)
        x = self.bn1(x) # shape = (1,16,15,15)
        x = self.activation(x) # shape = (1,16,15,15)
        if is_training: x = self.dropoutcnn(x)

        x = self.conv2(x) # shape = (1,32,13,13)
        x = self.padd2(x) # shape = (1,32,6,6)
        x = self.bn2(x)
        x = self.activation(x) # shape = (1,32,6,6)
        if is_training: x = self.dropoutcnn(x)

        x = self.conv3(x) # shape = (1,64,4,4)
        x = self.padd3(x) # shape = (1,64,2,2)
        x = self.bn3(x) # shape = (1,64,2,2)
        x = self.activation(x) # shape = (1,64,2,2)  
        if is_training: x = self.dropoutcnn(x)

        x = self.flatten(x) # shape (1,256)

        x = self.linear1(x) # shape = (1,64)

        if is_training: x = self.dropout1(x)

        x = self.activation(x) # shape = (1,64)

        x = self.linear2(x) # shape = (1,64)
        if is_training: x = self.dropout1(x)
        x = self.activation(x) # shape = (1,64)

        x = self.linear3(x) # shape = (1,19)
        return x
    
    def accuracy(self,
                 predicts: torch.tensor,
                 labels: torch.tensor
                 ) -> torch.float:
        """Get accuracy of model"""
        predicts = torch.argmax(predicts, dim=1)
        labels = torch.argmax(labels, dim=1)
        return torch.sum(predicts.eq(labels).float())
    
    def confusion_matrix(self, 
                        validation_dataloader: torch.utils.data.DataLoader,
                        labels_dict: Dict[int, str]
                        ) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix for classification task"""

        confusion_matrix = {i:{c:0 for c in labels_dict.values()} for i in labels_dict.values()}

        with torch.no_grad():
            for data in validation_dataloader:
                vinputs, vlabels = data
                vpredicts = self.forward(vinputs)

                vlabels = torch.argmax(vlabels, dim=1)
                vpredicts = torch.argmax(vpredicts, dim=1)

                for vl, vp in zip(vlabels, vpredicts):
                    confusion_matrix[labels_dict[int(vl)]][labels_dict[int(vp)]] += 1 

        return confusion_matrix
    
    def fit(self,
            train_dataloader: torch.utils.data.DataLoader,
            validation_dataloader: torch.utils.data.DataLoader,
            learning_rate: float = 0.001,
            epochs: int = 5
            ) -> None:
        
        """Main training loop"""
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
    dataset = MathCharactersDataset("symbols")

    TRAIN_TEST_SPLIT = 0.8
    LEARNING_RATE = .01

    # Split datasets into train and validation sets
    train_size = int(TRAIN_TEST_SPLIT * len(dataset)) # train size = 8056
    validation_size = len(dataset) - train_size # validation_size = 2015
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=True)

    model = CharModel()
    #print(dataset[1002])
    #model.forward(train_dataset[1][0])
    #model.fit(train_dataloader,
    #        validation_dataloader,
    #        learning_rate=LEARNING_RATE,
    #        epochs=25)
    
    #con = model.confusion_matrix(validation_dataloader, dataset.labels)

    ## print confusion matrix in readable way
    #print("CONFUSION MATRIX")
    #print(u"\033[4m    \u2551" + u"\u2551".join(list(f"{z.center(4, " ")}" for z in con)) + u"\u2551 \033[0m")
    #for n in con:
    ##    tab = u"\u2551"
    #    for m in con[n].values():
    #        tab += str(m).center(4)
    #        tab += u"\u2551"
    #    print(n.ljust(3), tab)