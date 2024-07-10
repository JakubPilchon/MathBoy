import tkinter as tk
from tkinter import ttk


class Applicaion(tk.Tk):

    def __button_click(self):
        print(f"current color: {self.color.get()}")

    def __init__(self):
        super().__init__()
        
        #Options
        self.geometry("1250x750")
        self.title("MathBoy")

        #Variables
        self.color = tk.StringVar(self, "black")

        colorframe = ttk.Frame(self, height=55, width=300)
        colorframe.grid(sticky="NW")
        
        tk.Radiobutton(colorframe, variable = self.color, value="black", selectcolor="#000000", cursor="hand2").grid(column=0, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="white",  selectcolor="#ffffff", cursor="hand2").grid(column=0, row=1, sticky="NW")

        tk.Radiobutton(colorframe, variable = self.color, value="red",  selectcolor="#ff0000", cursor="hand2").grid(column=1, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="green",selectcolor="#00ff00", cursor="hand2").grid(column=1, row=1, sticky="NW")
        
        tk.Radiobutton(colorframe, variable = self.color, value="blue", selectcolor="#0000ff", cursor="hand2").grid(column=2, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="yellow",selectcolor="#ffee03", cursor="hand2").grid(column=2, row=1, sticky="NW")

        tk.Radiobutton(colorframe, variable = self.color, value="violet", selectcolor="#8503ff", cursor="hand2").grid(column=3, row=0, sticky="NW")
        tk.Radiobutton(colorframe, variable = self.color, value="pink", selectcolor="#ff03f2", cursor="hand2").grid(column=3, row=1, sticky="NW")

        button = ttk.Button(colorframe, text="Click button", command=self.__button_click)
        button.grid(column=4, row=1, sticky='NW')

        canvas = tk.Canvas(self, width=1250, height=700, background="#ffffff")
        canvas.grid(column=0, row=2)
        

if __name__ == "__main__":
    app = Applicaion()
    app.mainloop()