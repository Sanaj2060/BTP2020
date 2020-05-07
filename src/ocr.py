#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:40:09 2020

@author: jaoba
"""

import os
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter.ttk import *
from tkinter.messagebox import showinfo

filename=""

window = Tk()
'''style=ttk.Style()
style.theme_use('default')'''

'''Top Bar title'''
window.title("Tesseract for Manipuri Language")

'''label One'''
lblOne = Label(window, text="TESSERACT FOR MANIPURI LANGUAGE", font=("Arial Unicode MS", 25))
lblOne.grid(column=1, row=0, padx=10, pady=10)

'''file and destination selection'''
lblTwo = Label(window, text="Please select the image", font=(10))
lblTwo.grid(column=1, row=1, padx=10, pady=10)


'''Function'''
def selectBtnfunc():
    '''Image Selector'''
    global filename
    filename = askopenfilename()
    lblThree = Label(window, text=filename)
    lblThree.grid(column=1,row=3)
    

selectBtn = Button(window, text="Choose", command=selectBtnfunc)
selectBtn.grid(column=1, row=2, pady=10, padx=10)



def startOCR():
    output = (os.path.splitext(filename)[0])
    print(output)
    proc = os.system("tesseract "+filename+" "+output+" --oem 1 -l eng")
    if(proc==0):
        print("OCR Completed!!!")
        showinfo("Message","OCR Completed")
    
        
    

ocrBtn = Button(window, text="Start OCR", command=startOCR)
ocrBtn.grid(column=1, row=4, padx=10, pady=10)


'''main Window'''
window.geometry('520x300')
window.resizable(False, False)
window.mainloop()
