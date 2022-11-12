import cv2
import numpy as np
import sys
import json
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import time

def make_sharp_kernel(k: int):
  return np.array([
    [-k / 9, -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, k / 9],
    [-k / 9, -k / 9, -k / 9]
  ], np.float32)


class CompDetection:
    def __init__(self, isViz = False, tsize = 1000) -> None:
        self.m = MS_SSIM()
        self.tsize = tsize
        self.isViz = isViz

    def preprocess(self, img : np.ndarray) -> np.ndarray:
        imgh, imgw, _ = img.shape
        if imgh < 3000 and imgw < 3000:
            aspect = float(imgw/imgh)
            img = cv2.resize(img, dsize=(int(3000 * aspect), 3000), interpolation=cv2.INTER_CUBIC)       
        return img

    def sharpenImage(self, img : np.ndarray, k = 1) -> np.ndarray:
        """
        a image sharpenizer for low res scaled image
        """
        kernel = np.array([
            [-k / 9, -k / 9, -k / 9],
            [-k / 9, 1 + 8 * k / 9, k / 9],
            [-k / 9, -k / 9, -k / 9]
        ], np.float32)
        img = cv2.filter2D(img, -1, kernel)
        return img

    def binarize(self, img: np.ndarray, thres = 185) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bin = cv2.threshold(gray, thres, 255,cv2.THRESH_BINARY)
        bin = cv2.dilate(bin, (4,4), iterations = 1)
        return bin

    def detectHeroRegion(self, img : np.ndarray, isSharpen = False) -> dict:
        imgh, imgw, _ = img.shape
        img = self.preprocess(img)
        if imgh < 3000 and imgw < 3000 and isSharpen:
            img = self.sharpenImage(img)

        
        bin = self.binarize(img)

        contours, hierachy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        x,y, w,h = 0,0,0,0
        maxarea = 0
        imgh, imgw, _ = img.shape
        """
        Crop to Result yellow paper
        """
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:

                area =cv2.contourArea(contours[i])
                if float(area/(imgw*imgh)) < 0.4:
                    continue

                if area > maxarea:
                    maxarea = area
                    rect = contours[i]
                    x, y, w, h = cv2.boundingRect(rect)

        cropped = bin[y : y+h, x:x+w]
        contours, hierachy = cv2.findContours(cropped, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        bboxs = []
        cnt = 0
        charimg = dict()
        charbbox = dict()
        cntimg = img.copy()
        """
        Get Character Images
        """
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                

                area =cv2.contourArea(contours[i])
                """
                Remove big contours and small contours
                """
                if float(area/(imgw*imgh)) > 0.7:
                    continue

                elif float(area/(imgw*imgh)) < 0.00025:
                    continue

                rect = contours[i]
                x2, y2, w2, h2 = cv2.boundingRect(rect)

                """
                remove nonsquare boundingbox
                """
                if abs(h2/w2) < 0.8 or abs(h2/w2) > 1.2:
                    continue

                elif float(h2/w2) < 0.2:
                    continue 
            
                """
                Remove boundingbox which is placed right of the screen
                """
                if float((x+x2+w2)/imgw) > 0.3:
                    continue
                """
                Remove boudingbox which is placed top of the screen
                """
                if float((y+y2+h2)/imgh) < 0.32:
                    continue

                hparam = 0.53 if imgw/imgh < 0.5 else 0.57
                """
                Remove boundingbox which is placed half bottom of the screen
                """
                if float((y+y2+h2)/imgh) > hparam:
                    continue
            
                bbox = {"x" : x+x2 + w2/2 ,"y" : y+y2 + h2/2, "width" : w2, "height" : h2}
                """
                Remove duplicated boundingbox which detect the same object
                """
                #arearat = w2 * h2/(imgw*imgh)
                # = int(0.0025 * imgh) if (isSharpen and arearat >= 0.0035) else 0
                #print("area : {0}, {1}".format(w2 * h2/(imgw*imgh), (w2-margin) * (h2 -margin)/(imgw*imgh)))
                #cv2.rectangle(cntimg, (x+x2 + margin/2, y+y2 + margin/2), (x+x2 + w2 - margin/2, y+y2 + h2 - margin/2), (0, 0, 255), 1)
                if len(bboxs) > 0:
                    if abs(bbox["x"] - bboxs[len(bboxs) - 1]["x"]) < 10 and abs(bbox["y"] - bboxs[len(bboxs) - 1]["y"]) < 10:
                        continue

                cnt += 1
                bboxs.append(bbox)
                #charimg["dps_{0}".format(cnt)] = img[y+y2:y+y2+h2, x+x2:x+x2+w2]

                """
                crop based on image matching best position
                """
                #charimg["dps_{0}".format(cnt)] = cv2.resize(charimg["dps_{0}".format(cnt)], dsize=(self.tsize, self.tsize), interpolation=cv2.INTER_CUBIC)
                #bh = charimg["dps_{0}".format(cnt)].shape[0]
                #bw = charimg["dps_{0}".format(cnt)].shape[1]
                charbbox["dps_{0}".format(6-cnt)] = {"x1" : x+x2, "x2" : x+x2+w2, "y1" : y+y2, "y2" : y + y2 + h2}
                cv2.rectangle(img, (x+x2, y+y2), (x+x2 + w2, y+y2 + h2), (0, 0, 255), 1)
        charbbox = dict(sorted(charbbox.items()))

        cv2.imwrite("bbox.png", img)
        #v2.imwrite("cnt.png", img)
        return charbbox

    def extractHeroRegion(self, img : np.ndarray, charbbox : dict):
        charimg = dict()
        for key, item in charbbox.items():
            charimg[key] = img[charbbox[key]["y1"]:charbbox[key]["y2"], 
                                charbbox[key]["x1"]:charbbox[key]["x2"]]

        return charimg


    def preprocessHeroDetection(self, charimg : dict) -> dict:
        for key, item in charimg.items():
            charimg[key] = cv2.resize(charimg[key], dsize=(self.tsize, self.tsize), interpolation=cv2.INTER_CUBIC)
            crop = self.cropHeroImages(charimg[key])
            charimg[key] = charimg[key][crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]

        return charimg
        
    def cropHeroImages(self, img, x1 = 0.3, x2 = 0.2, y1 = 0.3, y2 = 0.3) -> tuple:
        h,w, _ = img.shape
        x1 = int(w * x1)
        x2 = int(w * (1-x2))
        y1 = int(h * y1)
        y2 = int(h * (1-y2))
        return {"x1" : x1,  "x2" : x2, "y1" : y1, "y2" : y2}

    def makeHeroDict(self, config_file="char_images_new/char_image.json") -> dict: 
        charimg_dict = dict()
        f = open(config_file, "r")
        pathdict = json.load(f)
        for key, item in pathdict.items():
            charimg_dict[key] = cv2.imread("char_images_new/{0}".format(item.replace(".jpg", "_dead.jpg")))
            charimg_dict[key] = cv2.resize(charimg_dict[key], dsize=(self.tsize, self.tsize), interpolation=cv2.INTER_CUBIC)
            crop = self.cropHeroImages(charimg_dict[key])
            charimg_dict[key] = charimg_dict[key][crop["y1"]:crop["y2"], crop["x1"] : crop["x2"]]
        return charimg_dict

    def detectHero(self, img : np.ndarray, charbbox : dict, herodict : dict) -> dict:
        img = self.preprocess(img)
        charimg = self.extractHeroRegion(img, charbbox)
        charimg = self.preprocessHeroDetection(charimg)
        #print(charimg)
        #print(charimg["dps_1"].shape)
        #print(herodict["belinda"].shape)
        comps = dict()

        for key, item in charimg.items():
            img1 = torch.from_numpy(np.rollaxis(charimg[key], 2)).float().unsqueeze(0)/255.
            score = dict()
            for char, data in herodict.items():
                img2 = torch.from_numpy(np.rollaxis(herodict[char], 2)).float().unsqueeze(0)/255.0
                ssim_loss = self.m(img1, img2)
                score[char] = ssim_loss.item()
            max_k = max(score, key=score.get)
            score = sorted(score.items(), key = lambda x : x[1], reverse=True)
            comps[key] = max_k.replace("2", "")
            print(score[:10])
            if self.isViz:
                cv2.imshow("real", charimg[key])
                cv2.imshow("predict", herodict[max_k])
                cv2.waitKey(1000)
            

        return comps

    def detect(self, img: np.ndarray):
        charbbox = self.detectHeroRegion(img)
        herodict = self.makeHeroDict()
        comps = self.detectHero(img, charbbox, herodict)
        return comps
"""
if __name__ == "__main__":
    path = sys.argv[1]
    img = cv2.imread(path)
    compdetect = CompDetection()
    comp = compdetect.detect(img)
    print(comp)
"""

if __name__ == "__main__":
    path = sys.argv[1]
    if len(sys.argv) > 2:
        path = sys.argv[1]
        img = cv2.imread(path)
        compdetect = CompDetection()
        comp = compdetect.detect(img)
        print(comp)
"""
    img = cv2.imread(path)
    imgh, imgw, _ = img.shape
    if imgh < 3000 and imgw < 3000:
        aspect = float(imgw/imgh)
        img = cv2.resize(img, dsize=(int(3000 * aspect), 3000), interpolation=cv2.INTER_CUBIC)
        imgh, imgw, _ = img.shape
        kernel = make_sharp_kernel(2)
        #img = cv2.filter2D(img, -1, kernel).astype("uint8")

    #img = cv2.resize(img, dsize=(w*5, h*5), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 185, 255,cv2.THRESH_BINARY)
    
    #bin = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    bin = cv2.dilate(bin, (4,4), iterations = 1)

    contours, hierarchy = cv2.findContours(
        bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours, hierachy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    x,y, w,h = 0,0,0,0
    maxarea = 0

    for i in range(0, len(contours)):
        if len(contours[i]) > 0:

            # remove small objects
            #if cv2.contourArea(contours[i]) < 500:
            area =cv2.contourArea(contours[i])
            if float(area/(imgw*imgh)) < 0.:
                continue

            if area > maxarea:
                maxarea = area
                print(maxarea)
            #continue
            #cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)
                rect = contours[i]
                x, y, w, h = cv2.boundingRect(rect)

            #rect2 = contours[i]
            #x2, y2, w2, h2 = cv2.boundingRect(rect2)
            #cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 0), 10)
            #cv2.polylines(img, contours[i], True, (255, 255, 255), 5)

    cropped = bin[y : y+h, x:x+w]
    contours, hierachy = cv2.findContours(cropped, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxs = []
    cnt = 0

    charimg = dict()

    for i in range(0, len(contours)):
        if len(contours[i]) > 0:

            # remove small objects
            #if cv2.contourArea(contours[i]) < 500:
            area =cv2.contourArea(contours[i])
            if float(area/(imgw*imgh)) > 0.7:
                continue

            elif float(area/(imgw*imgh)) < 0.0001:
                continue

            #continue
            #cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)
            rect = contours[i]
            x2, y2, w2, h2 = cv2.boundingRect(rect)
            if abs(h2/w2) < 0.8 or abs(h2/w2) > 1.2:
                continue
            elif float(h2/w2) < 0.2:
                continue 
            
            if float((x+x2+w2)/imgw) > 0.3:
                continue

            #if float((x+x2 + w2)/imgw) > 0.5:
            #    continue

            if float((y+y2+h2)/imgh) < 0.32:
                continue

            hparam = 0.53 if imgw/imgh < 0.5 else 0.57
            if float((y+y2+h2)/imgh) > hparam:
                continue
            
            
            print(imgw/imgh)
            #margin = int(float(imgw * 0.01))
            margin = 0
            print("margin : {0}".format(margin))
            
            
            bbox = {"x" : x+x2 + (w2 + margin)/2 ,"y" : y+y2 + (h2+margin)/2, "width" : w2 + margin * 2, "height" : h2 + margin * 2}
            print(bbox)

            if len(bboxs) > 0:
                if abs(bbox["x"] - bboxs[len(bboxs) - 1]["x"]) < 10 and abs(bbox["y"] - bboxs[len(bboxs) - 1]["y"]) < 10:
                    continue

            cnt += 1
            bboxs.append(bbox)
            cv2.rectangle(img, (x+x2 - margin, y+y2 - margin), (x+x2 + w2 + margin, y+y2 + h2 + margin), (0, 0, 255), 1)
            charimg["dps_{0}".format(cnt)] = img[y+y2:y+y2+h2, x+x2:x+x2+w2]
            charimg["dps_{0}".format(cnt)] = cv2.resize(charimg["dps_{0}".format(cnt)], dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)
            bh = charimg["dps_{0}".format(cnt)].shape[0]
            bw = charimg["dps_{0}".format(cnt)].shape[1]
            x1croprat = 0.3
            x2croprat = 0.3
            y1croprat = 0.3
            y2croprat = 0.2
            charimg["dps_{0}".format(cnt)] = charimg["dps_{0}".format(cnt)][int(bh * x1croprat) : int(bh * (1-x2croprat)), int(bw * y1croprat): int(bw * (1-y2croprat))]


    bboxs = list(reversed(bboxs))
    mergedbbox = []
    mIdx = 0

    cv2.imwrite("cropped.png", cropped)
    cv2.imwrite("bbox.png", img)
    cv2.imwrite("bin.png", bin)

    charimg_dict = dict()
    f = open("char_images_new/char_image.json", "r")
    pathdict = json.load(f)
    for key, item in pathdict.items():
        charimg_dict[key] = cv2.imread("char_images_new/{0}".format(item.replace(".jpg", "_dead.jpg")))
        #charimg_dict[key] = cv2.imread("char_images_new/{0}".format(item))
        print("char_images_new/{0}".format(item))
        charimg_dict[key] = cv2.resize(charimg_dict[key], dsize=(1000, 1000))
        bh = charimg_dict[key].shape[0]
        bw = charimg_dict[key].shape[1]
        x1croprat = 0.3
        x2croprat = 0.3
        y1croprat = 0.3
        y2croprat = 0.2
        charimg_dict[key] = charimg_dict[key][int(bh * x1croprat) : int(bh * (1-x2croprat)), int(bw * y1croprat): int(bw * (1-y2croprat))]

    #print(charimg)
    #print(charimg_dict)

    m = MS_SSIM()
    #print(len(charimg))

    for key, item in charimg.items():
        img1 = torch.from_numpy(np.rollaxis(charimg[key], 2)).float().unsqueeze(0)/255.
        score = dict()
        for char, data in charimg_dict.items():
            img2 = torch.from_numpy(np.rollaxis(charimg_dict[char], 2)).float().unsqueeze(0)/255.0
            #print(char)
            #(img1.size())
            #print(img2.size())
            ssim_loss = m(img1, img2)
            score[char] = ssim_loss.item()
        max_k = max(score, key=score.get)
        #print(max_k)
        score = sorted(score.items(), key = lambda x : x[1], reverse=True)
        print(score[:10])
        cv2.imshow("real", charimg[key])
        cv2.imshow("predict", charimg_dict[max_k])
        cv2.waitKey(10)

    time.sleep(1)
"""
"""
    for i in range(len(bboxs)):
        if i > 0:
            x1start = bboxs[i]["x"] - bboxs[i]["width"]/2
            x1end = bboxs[i]["x"] + bboxs[i]["width"]/2
            y1start = bboxs[i]["y"] - bboxs[i]["height"]/2
            y1end = bboxs[i]["y"] + bboxs[i]["height"]/2

            x2start = bboxs[mIdx]["x"] - bboxs[mIdx]["width"]/2
            x2end = bboxs[mIdx]["x"] + bboxs[mIdx]["width"]/2
            y2start = bboxs[mIdx]["y"] - bboxs[mIdx]["height"]/2
            y2end = bboxs[mIdx]["y"] + bboxs[mIdx]["height"]/2
            print("x1 : [{0}, {1}], y1 : [{2}, {3}]".format(x1start, x1end, y1start, y1end))
            print("x2 : [{0}, {1}], y2 : [{2}, {3}]".format(x2start, x2end, y2start, y2end))        
            xOverlaps = ((x2start >= x1start) and (x1end >= x2start)) or ((x2end >= x1start) and (x1end >= x2end))
            yOverlaps = ((y2start >= y1start) and (y1end >= y2start)) or ((y2end >= y2start) and (y1end >= y2end))
            if xOverlaps or yOverlaps:
                bboxs[mIdx]["x"] = (bboxs[mIdx]["x"] + bboxs[i]["x"])/2
                widthOffset = (bboxs[i]["x"] - bboxs[i]["width"]/2) - (bboxs[mIdx]["x"] - bboxs[mIdx]["width"]/2)
                widthOffset = 0 if widthOffset < 0 else widthOffset
                bboxs[mIdx]["width"] = max(bboxs[mIdx]["width"], bboxs[i]["width"]) + widthOffset

                bboxs[mIdx]["y"] = (bboxs[mIdx]["y"] + bboxs[i]["y"])/2
                heightOffset = (bboxs[i]["y"] - bboxs[i]["height"]/2) - (bboxs[mIdx]["y"] - bboxs[mIdx]["height"]/2)
                heightOffset = 0 if heightOffset < 0 else heightOffset
                bboxs[mIdx]["height"] = max(bboxs[mIdx]["height"], bboxs[i]["height"]) + heightOffset
            else:
                mergedbbox.append(bboxs[mIdx])
                mIdx = i -1


    for i in range(len(mergedbbox)):
        cv2.rectangle(img, (int(mergedbbox[i]["x"] - mergedbbox[i]["width"]/2), int(mergedbbox[i]["y"] - mergedbbox[i]["height"]/2)), 
        (int(mergedbbox[i]["x"] + mergedbbox[i]["width"]/2), int(mergedbbox[i]["y"] + mergedbbox[i]["height"]/2)), (0, 0, 0), 10)
"""

"""
    dmgdata = dict()
    #bboxs = list(reversed(bboxs))
    annot = []
    for i in range(len(bboxs)):
        annot.append({"label" :  "dps_{0}".format(i+1), "coordinates" :bboxs[i]})

    dmgdata["annotations"] = annot
    print(dmgdata)
    f = open("dmg_dataex.json", "w+")
    json.dump([dmgdata], f)
"""
    #cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)

    #cv2.imshow("contour", img)
    
    #cv2.imshow("bin", bin)
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()