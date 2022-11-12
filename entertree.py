import sys
from tkinter import *
from PIL import ImageTk, Image  
import os

if __name__ == "__main__":
    path = sys.argv[1]
    dirs = sorted(os.listdir(path))
    
    root = Tk()

    dirid = 0
    