import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk,Image  

class GAN_Demo:
    def __init__(self):
        self.root = tk.Tk()  

    def refresh(self, I):
        for widget in self.root.winfo_children():
            widget.destroy()

        epoch = tk.Label(self.root, text = f"Epoch: {I}")
        # epoch.grid(row=0, column=0, sticky=tk.W) 
        epoch.pack()

        canvas = tk.Canvas(self.root, width = 400, height = 400)  
        canvas.pack()  
        try:
            img = ImageTk.PhotoImage(Image.open("images/demo.png").resize((400, 400)))  
        except Exception:
            img = ImageTk.PhotoImage(Image.open("cgize/images/demo.png").resize((400, 400)))  

        canvas.create_image(0, 0, anchor=tk.NW, image=img) 

        self.root.mainloop() 


