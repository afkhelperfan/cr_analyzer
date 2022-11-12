import sys
import os
import math
import shutil

if __name__ == "__main__":
    path = sys.argv[1]
    imgs = os.listdir(path)
    imgs = list(filter(lambda x: x.find(".png") != -1, imgs))
    imgs = sorted(imgs)
    trial = int(len(imgs)/6)
    for i in range(trial):
        os.mkdir(path+"\\{0}".format(i+1))
    
    for i in range(len(imgs)):
        trial = math.floor(i/6) + 1
        comp = i % 6 + 1
        os.rename("{0}\\{1}".format(path, imgs[i]), "{0}\\{1}\\{2}.png".format(path, trial, comp))


