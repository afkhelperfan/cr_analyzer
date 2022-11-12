import os
import sys
import cv2

if __name__ == "__main__":

    if len(sys.argv) < 1:
        print("specify path")

    path = sys.argv[1]

    dirs = sorted(os.listdir(path))
    #print("{0} dirs are reversed".format(len(dirs)))

    for dir in dirs:
        if dir.find("reversed") != -1:
            images = sorted(os.listdir("{0}/{1}".format(path, dir)), key=lambda x : int(x.split(".")[0]))
            print(images)
            
            imgs = []
            ext = ""
            
            #load all images
            for i in range(len(images)):
                imgs.append(cv2.imread("{0}/{1}/{2}".format(path, dir, images[i])))
                ext = images[i].split(".")[1]

            #reverse it
            for i in range(len(imgs)):
                reverseidx = len(imgs) - i
                print("{0}/{1}/{2}.{3}".format(path, dir, reverseidx, ext))
                cv2.imwrite("{0}/{1}/{2}.{3}".format(path, dir, reverseidx, ext), imgs[i])
                #print("replaced {0}".format(images[i]))

            print("replacing dir name {0}".format(dir))
            os.rename("{0}/{1}".format(path, dir), "{0}/{1}".format(path, dir).replace("-reversed", ""))

    dirs = sorted(os.listdir(path))

    for dir in dirs:
        if dir.find("dupe") != -1:
            images = sorted(os.listdir("{0}/{1}".format(path, dir)), key=lambda x : int(x.split(".")[0]))
            print(images)
            
            #load all images
            for i in range(len(images)):
                img = cv2.imread("{0}/{1}/{2}".format(path, dir, images[i]))
                ext = images[i].split(".")[1]
                cv2.imwrite("{0}/{1}/{2}.{3}".format(path, dir, i+1, ext), img)

            print("replacing dir name {0}".format(dir))
            os.rename("{0}/{1}".format(path, dir), "{0}/{1}".format(path, dir).replace("-dupe", ""))
            
                