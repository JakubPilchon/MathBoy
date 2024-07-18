import torch
import os
import cv2 as cv
import numpy as np
## set device

class MathCharactersDataset(torch.utils.data.Dataset):
    def __init__(self, dir:str):
        assert os.path.isdir(dir)
        super().__init__()
        self.dir = dir
        self.data = []
        ## get dataset annotations
        for class_name in os.listdir(self.dir):
            for data_name in os.listdir(os.path.join(self.dir, class_name)):
                self.data.append((os.path.join(self.dir, class_name, data_name),class_name ))


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]




if __name__ == "__main__":
    dataset = MathCharactersDataset("dataset")
    #print(dataset.data[])
    print(len(dataset))
    print(dataset[2:5])

        
        





