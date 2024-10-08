import cv2 as cv
import os
import torch
#from torchvision.transforms import Grayscale, RandomRotation, Compose, ConvertImageDt
from torchvision.transforms import v2
from dataclasses import dataclass
from typing import List, Tuple, NewType, Dict
from model_train.train import CharModel
from PIL import Image, ImageChops
import numpy as np

## custom types:
Bounding_boxes = NewType("Bounding_boxes", List[Tuple[int,int,int,int]])
Pillow_image = NewType("Pillow_image", Image.Image)

@dataclass
class Character:
    x: int
    y: int
    w: int
    h: int
    label: str

    def get_points(self) -> Tuple[int,int,int,int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)
    
    def __repr__(self) -> str:
        return f'<character: "{self.label}" at ({self.x}, {self.y}, {self.w}, {self.h})>'


class Preprocessor:
    
    def __init__(self):
        #Constants:
        self.KERNEL = (3,9)

        self.model = CharModel()
        self.model.load_state_dict(torch.load("mathboy/model.pt"))
        self.model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model.to(self.device)

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(),
            v2.Resize((28,28)),
            v2.Lambda(v2.functional.invert)
        ])

        self.classes_lookup: Dict[int, str] = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "*", 11: "=", 12: "-", 13: "+",
            14: "/", 15: "w", 16: "x", 17: "y", 18: "z"}

    def get_picture(self):
        try:
            img = cv.imread("img.jpg")
            
        except Exception as e:
            print(e)
        
        return img
    
    def get_bounding_boxes(self, image: Pillow_image) -> Bounding_boxes:
        """detect characters in image and construct bounding boxes around characters. Return as (x,y,w,h)"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Convert image to Grayscale
        img_pre = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Thresholding technique on image, makes every character black
        _, img_pre = cv.threshold(img_pre, 240, 255, cv.THRESH_OTSU)
        # Crops image, because loaded up image is 4 pixels higher and wider than it should be
        
        # Reverse colors, now characters are white and background is black
        img_pre = cv.bitwise_not(img_pre)

        # makes characters bigger, better for finding countours
        dilated = cv.dilate(img_pre, cv.getStructuringElement(cv.MORPH_RECT, self.KERNEL), iterations=4)

        #finds contours of characters
        contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for n in contours:
            # finds rectange encompassing character countour
            x,y,w,h = cv.boundingRect(n)
            if w > 30:
                rectangles.append((x,y,w,h))
        
        return rectangles   

    def get_characters(self, image: Pillow_image, bounding_boxes:Bounding_boxes) -> List[Character]:
        """Classify characters and return them as list of Character objects."""

        character_list = []
        for (x, y, w, h) in bounding_boxes:
            character = image.crop((x, y, x+w, y+h))
            character = self.transforms(character).to(self.device)
            char_class = self.model.forward(character)
            char_class = self.classes_lookup[int(torch.argmax(char_class))]

            character_list.append(Character(x,y,w,h, char_class))

        return character_list
    
    def cluster_datatset(self, characters: List[Character]) -> List[List[Character]]:
        """Organize characters into mathematical expressions"""
        clusters = []

        mean_h = sum([char.h for char in characters]) / len(characters)
        characters.sort(key=lambda char: char.y)

        while characters:
            stack = []
            char = characters.pop(0)
            stack.append(char)
            for c in stack:
                if characters:
                    if (characters[0].y <= (c.y + mean_h)):
                        stack.append(characters.pop(0)) 
                else:
                    break
            stack.sort(key=lambda char: char.x, reverse=False)
            clusters.append(stack)

        return clusters
    
    def solve(self, expressions: List[List[Character]]) -> List[Tuple[int, int, int, int]]:
        """Read and solve mathematical expressions, return values and info"""
        answers: List[Tuple[str, int, int, int]] = []
        variables = {"x": '', "y": '', "z": '', "w":''}

        non_var: List[List[Character]] = []
        #Read variables assignments
        for exp in expressions:
            no_error = True     
            exp_text = ''.join(char.label for char in exp)
            if self.__check_num_of_letter_instances(exp_text, "=") == 1 and exp_text.split("=")[0] in ["x", "y", "z", "w"]:
                exp_text = exp_text.split("=")
                try:
                    variables[exp_text[0]] = str(eval(exp_text[1])) # assign variable
                except Exception:
                    no_error = False
                    answers.append(self.__return_error(exp))
            else:
                non_var.append(exp)

        # check for expressions to calculate
        for exp in non_var:

            exp_text = ''.join(char.label for char in exp)

            for var in variables: # replace letters with variable values
                    exp_text = exp_text.replace(var, variables[var])

            if self.__check_num_of_letter_instances(exp_text, "=") == 0:   
                try:
                    evaluated = eval(exp_text)
                    if isinstance(evaluated, float):
                        evaluated = round(evaluated, 3)
                    evaluated = "=" + str(evaluated)
                    print("hi", evaluated)
                except Exception:
                    no_error = False
                    answers.append(self.__return_error(exp))

            elif self.__check_num_of_letter_instances(exp_text, "=") == 1:
                try:
                    exp_text = exp_text.replace("=", "==")
                    evaluated = eval(exp_text)
                    evaluated = ":" + str(evaluated)
                    print(evaluated)
                except Exception:
                    no_error = False
                    answers.append(self.__return_error(exp))
                      
            if no_error:
                answers.append((evaluated, # evaluated answer
                                int(sum([char.y for char in exp])/len(exp) + exp[-1].h/2), # y position of answer
                                exp[-1].x + int(len(evaluated)/2 * exp[-1].w) + 100, # x position of answer
                                self.__px_to_pt(px = exp[-1].h),# height of answer in points
                                0)) # width (not neccessary in this case)       

        return answers      

    ## Interal functions:
    #  
    def __return_error(self, expression: List[Character]) -> Tuple[str, int, int, int, int]:
        """Transforms expressoin into response format.
        Response format: {response_text, y,x,h,w}"""
        x = expression[0].x # x position of first character in expression
        y = min(*(char.y for char in expression)) # y is average y of given expresson
        h = int(max(*(char.h for char in expression))) # height is maximum height of an character in expression
        w = expression[-1].x - expression[0].x + expression[-1].w
        return ("ERROR", y,x,h,w)

    def __check_num_of_letter_instances(self, string:str, char:str) -> int:
        """Calculate number if instances of letter in string sequence"""
        i = 0
        for let in string:
            if let == char: i+=1
        return i
    
    def __px_to_pt(self, px:int | float) -> int:
        """Convert height in pixels to points"""
        return int(px * 72/96)

if __name__ == "__main__":
    pre = Preprocessor()

    y_values = [386, 347, 336, 311, 87, 59, 58, 45, 40, 35, 33]

    characters = [
        Character(x=i, y=y, w=10 + i, h=20 + i, label=f"Label_{i}")
        for i, y in enumerate(y_values)]

    print(pre.cluster_datatset(characters))
