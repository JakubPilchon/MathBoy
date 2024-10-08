import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab
from PIL import Image
from preprocessor import Preprocessor
import pyscreenshot
import platform


class Application(tk.Tk):

    def __button_click(self) -> None:

        img = self.__get_picture()
        chars = self.preprocessor.get_characters(img, self.preprocessor.get_bounding_boxes(img))
        clusters = self.preprocessor.cluster_datatset(chars)

        if self.verbose:
            colors = ["red", "green","blue", "yellow", "black", "pink", "purple"]
            for i, cluster in enumerate(clusters):
                text = ""
                print(f"cluster {i}")
                for c in cluster:
                    self.canvas.create_rectangle(((c.x, c.y), (c.x + c.w, c.y+ c.h)))
                    self.canvas.create_text(c.x, c.y+10, text=c.label,  font=('Helvetica 24 bold'), fill=colors[i])
                    text += c.label
                    print("   ", c)
                print("Expression: ", {text})

        answers = self.preprocessor.solve(clusters)
        if answers:
            for (solved, y, x, h, w) in answers:
                if solved != "ERROR":
                    self.canvas.create_text(x+30,y, text=str(solved),  font=(self.font, h))
                else:
                    self.canvas.create_rectangle(((x,y+h), (x+w, y)), outline="red", tags="error_mes")
                    self.canvas.create_text(x,y - 20, text="error", fill="red", tags="error_mes")

                    self.error_button["state"] = "normal"

    def __paint(self, event) -> None:
        """creates oval in place where mouse is."""
        if self.is_painting:
            self.canvas.create_oval(event.x- self.scale.get(),
                                    event.y-self.scale.get(),
                                    event.x+self.scale.get(),
                                    event.y+self.scale.get(),
                                    fill=self.color.get(),
                                     width=0)

    def __start_paint(self, event) -> None:
        self.is_painting = True

    def __stop_paint(self, event) -> None:
        self.is_painting = False
    
    def __reset(self) -> None:
        """deletes all painting from canvas"""
        self.canvas.delete("all")

    def __get_picture(self) -> Image.Image:
        """Screenshots Canvas in order to save is as bitmap image."""
        # canvas coordinates
        x1 = self.winfo_rootx() + self.canvas.winfo_x()
        y1 = self.winfo_rooty() + self.canvas.winfo_y()

        x2 = x1 + self.canvas.winfo_width()
        y2 = y1 + self.canvas.winfo_height()
        
        if platform.system() == "Linux": # handle diffrent systems
            img = pyscreenshot.grab((x1,y1,x2,y2))
            img = img.convert("RGB")
        else:
            img = ImageGrab.grab((x1,y1,x2,y2), all_screens=True)

        img = img.crop((0,0,1249, 670))
        return img
    
    def __delete_errors(self) -> None:
        self.canvas.delete("error_mes")
        self.error_button["state"] = "disabled"

    def __init__(self, verbose:bool = False):
        super().__init__()

        self.preprocessor = Preprocessor()
        
        # Options
        self.geometry("1250x750")
        self.maxsize(1250,750)
        self.title("MathBoy")

        # Variables
        self.color = tk.StringVar(self, "black")
        self.scale = tk.IntVar(self, 5)
        self.font = "Segoe Script" if platform.system() != "Linux" else "Z003"
        self.verbose = verbose

        # Define app layout
        colorframe = ttk.Frame(self, height=55, width=300, borderwidth=3)
        colorframe.grid(sticky="NW")
        
        tk.Radiobutton(colorframe, variable = self.color, value="black",  selectcolor="#000000", cursor="hand2").grid(column=0, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="white",  selectcolor="#ffffff", cursor="hand2").grid(column=0, row=1, sticky="NW")

        tk.Radiobutton(colorframe, variable = self.color, value="red",  selectcolor="#ff0000", cursor="hand2").grid(column=1, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="green",selectcolor="#00ff00", cursor="hand2").grid(column=1, row=1, sticky="NW")
        
        tk.Radiobutton(colorframe, variable = self.color, value="blue",  selectcolor="#0000ff", cursor="hand2").grid(column=2, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="#e68f0e",selectcolor="#e68f0e", cursor="hand2").grid(column=2, row=1, sticky="NW")

        tk.Radiobutton(colorframe, variable = self.color, value="purple", selectcolor="#8503ff", cursor="hand2").grid(column=3, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="#de09cc",   selectcolor="#ff03f2", cursor="hand2").grid(column=3, row=1, sticky="NW")

        button = ttk.Button(colorframe, text="Solve", command=self.__button_click)
        button.grid(column=4, row=1, sticky='NW')

        reset_button = ttk.Button(colorframe, text="Reset", command= self.__reset)
        reset_button.grid(column=4, row=0, sticky='NW')

        self.error_button = ttk.Button(colorframe, text="Delete Errors", state="disabled", command= self.__delete_errors)
        self.error_button.grid(column=5, row=0)

        scale = tk.Scale(colorframe,  variable=self.scale, from_=2, to=15, orient=tk.HORIZONTAL)
        scale.grid(column=6, row=0, rowspan=2, columnspan=2)

        self.canvas = tk.Canvas(self, width=1250, height=700, background="#ffffff", cursor="plus")
        self.canvas.grid(column=0, row=2)


        ## Bind mouse
        self.canvas.bind("<B1-Motion>", self.__paint)
        self.canvas.bind("<Button-1>", self.__start_paint)
        self.canvas.bind("<ButtonRelease-1>", self.__stop_paint)
        

if __name__ == "__main__":
    app = Application(verbose=True)
    app.mainloop()