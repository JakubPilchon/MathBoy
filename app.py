import tkinter as tk
from tkinter import ttk


class Applicaion(tk.Tk):

    def __button_click(self):
        print(f"current color: {self.color.get()}")

    def __init__(self):
        super().__init__()
        self.color = tk.StringVar(self, "black")

        mainframe = ttk.Frame(self,padding=2, height=750, width=1250, border=10)
        #mainframe.grid()
        
    
        self.title("MathBoy")

        tk.Radiobutton(self, variable = self.color, value="black", selectcolor="#000000").grid(column=0, row=0, sticky="NW")
        tk.Radiobutton(self, variable = self.color, value="red",  selectcolor="#ff0000").grid(column=1, row=0, sticky="NW")
        tk.Radiobutton(self, variable = self.color, value="blue", selectcolor="#0000ff").grid(column=0, row=1, sticky="NW")
        tk.Radiobutton(self, variable = self.color, value="green",selectcolor="#00ff00").grid(column=1, row=1, sticky="NW")

        button = ttk.Button(self, text="Click button", command=self.__button_click).grid(column=2, row=1, sticky='NW')

        canvas = tk.Canvas(self, width=1250, height=700, background="#ffffff")
        canvas.grid(column=0, row=2)
        

if __name__ == "__main__":
    app = Applicaion()
    app.mainloop()