import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab
from PIL import Image
from preprocessor import Preprocessor


class Applicaion(tk.Tk):

    def __button_click(self) -> None:
        img = self.__get_picture()
        #boxes = self.preprocessor.get_bounding_boxes(img)
        chars = self.preprocessor.get_characters(img, self.preprocessor.get_bounding_boxes(img))
        colors = ["red", "green","blue", "yellow", "black", "pink", "purple"]
        clusters = self.preprocessor.cluster_datatset(chars)
        #print(clusters)
        for i, cluster in enumerate(clusters):
            #color = colors[i]
            print(f"cluster {i}")
            for c in cluster:
                self.canvas.create_rectangle(((c.x, c.y), (c.x + c.w, c.y+ c.h)))
                self.canvas.create_text(c.x, c.y+10, text=c.label,  font=('Helvetica 24 bold'), fill=colors[i])
                print("   ", c)

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
        
        img = ImageGrab.grab((x1,y1,x2,y2), all_screens=True)
        #img = img[:689, :1249]
        img = img.crop((0,0,1249, 689))
        return img

    def __init__(self):
        super().__init__()

        self.preprocessor = Preprocessor()
        
        # Options
        self.geometry("1250x750")
        self.maxsize(1250,750)
        self.title("MathBoy")

        # Variables
        self.color = tk.StringVar(self, "black")
        self.scale = tk.IntVar(self, 5)

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

        button = ttk.Button(colorframe, text="Click button", command=self.__button_click)
        button.grid(column=4, row=1, sticky='NW')

        reset_button = ttk.Button(colorframe, text="reset", command= self.__reset)
        reset_button.grid(column=4, row=0, sticky='NW')

        scale = tk.Scale(colorframe,  variable=self.scale, from_=2, to=15, orient=tk.HORIZONTAL)
        scale.grid(column=5, row=0, rowspan=2)

        self.canvas = tk.Canvas(self, width=1250, height=700, background="#ffffff", cursor="plus")
        self.canvas.grid(column=0, row=2)


        ## Bind mouse
        self.canvas.bind("<B1-Motion>", self.__paint)
        self.canvas.bind("<Button-1>", self.__start_paint)
        self.canvas.bind("<ButtonRelease-1>", self.__stop_paint)
        

if __name__ == "__main__":
    app = Applicaion()
    app.mainloop()