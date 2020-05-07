#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:40:09 2020

@author: jaoba
"""

import tkinter as tk
import os
from tkinter import *
from PIL import ImageTk, Image
from tkinter.messagebox import showinfo

window = tk.Tk()
'''style=ttk.Style()
style.theme_use('default')'''

'''Top Bar title'''
window.title("NMT (Bengali -> Meitei Mayek)")

img = ImageTk.PhotoImage(Image.open("src/assets/translator.png"))
panel = Label(window, image = img)
panel.grid(column=1, row = 1)




'''Function'''
def startTranslation():
    proc = os.system("python src/test.py -l ben-man")
    if(proc==0):
        print("Translation Completed!!!")
        showinfo("Message","Translation Completed")
    
        
    

Btn = Button(window, text="START", command=startTranslation)
Btn.grid(column=1, row=5, padx=10, pady=10)


'''main Window'''
window.geometry('670x500')
window.resizable(False, False)
window.mainloop()
