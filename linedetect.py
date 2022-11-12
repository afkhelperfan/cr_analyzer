import cv2
import numpy as np
import sys
import json


class StatDetector:
    def __init__(self) -> None:
        pass

    def detect(self, img):
        imgh, imgw, _ = img.shape
        #img = cv2.resize(img, dsize=(w*5, h*5), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bin = cv2.threshold(gray, 185, 255,cv2.THRESH_BINARY)

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
                    #print(maxarea)
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
                if h2/w2 > 1:
                    continue
                elif float(h2/w2) < 0.2:
                    continue 
            
                if float((x+x2+w2)/imgw) < 0.3:
                    continue

                if float((x+x2 + w2)/imgw) > 0.5:
                    continue

                

                if float((y+y2+h2)/imgh) < 0.32:
                    continue

                if float((y+y2+h2)/imgh) > 0.54:
                    continue
            
            
                margin = int(float(imgw * 0.01))
                #("margin : {0}".format(margin))
                cnt += 1
                bbox = {"x" : x+x2 + (w2 + margin)/2 ,"y" : y+y2 + (h2+margin)/2, "width" : w2 + margin * 2, "height" : h2 + margin * 2}
                #print(bbox)
                bboxs.append(bbox)
                cv2.rectangle(img, (x+x2 - margin, y+y2 - margin), (x+x2 + w2 + margin, y+y2 + h2 + margin), (0, 0, 0), 10)

        bboxs = list(reversed(bboxs))
        dmgdata = dict()
        #bboxs = list(reversed(bboxs))
        annot = []
        for i in range(len(bboxs)):
            annot.append({"label" :  "dps_{0}".format(i+1), "coordinates" :bboxs[i]})

        dmgdata["annotations"] = annot
        #print(dmgdata)
        cv2.imwrite("bbox-line.png", img)
        return [dmgdata]
        #f = open("dmg_dataex.json", "w+")
        #json.dump([dmgdata], f)

if __name__ == "__main__":
    path = sys.argv[1]
    img = cv2.imread(path)
    statDetector = StatDetector()
    dmgdata = statDetector.detect(img)
    print(dmgdata)
    """
    imgh, imgw, _ = img.shape
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
            if h2/w2 > 2:
                continue
            elif float(h2/w2) < 0.2:
                continue 
            
            if float((x+x2+w2)/imgw) < 0.3:
                continue

            if float((x+x2 + w2)/imgw) > 0.5:
                continue

            if float((y+y2+h2)/imgh) < 0.32:
                continue

            if float((y+y2+h2)/imgh) > 0.55:
                continue
            
            
            margin = int(float(imgw * 0.01))
            print("margin : {0}".format(margin))
            cnt += 1
            bbox = {"x" : x+x2 + (w2 + margin)/2 ,"y" : y+y2 + (h2+margin)/2, "width" : w2 + margin * 2, "height" : h2 + margin * 2}
            print(bbox)
            bboxs.append(bbox)
            cv2.rectangle(img, (x+x2 - margin, y+y2 - margin), (x+x2 + w2 + margin, y+y2 + h2 + margin), (0, 0, 0), 10)

    bboxs = list(reversed(bboxs))
    mergedbbox = []
    mIdx = 0
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
    """"""
    dmgdata = dict()
    #bboxs = list(reversed(bboxs))
    annot = []
    for i in range(len(bboxs)):
        annot.append({"label" :  "dps_{0}".format(i+1), "coordinates" :bboxs[i]})

    dmgdata["annotations"] = annot
    print(dmgdata)
    f = open("dmg_dataex.json", "w+")
    json.dump([dmgdata], f)

    #cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)
    cv2.imwrite("cropped.png", cropped)
    cv2.imwrite("bbox.png", img)
    cv2.imwrite("bin.png", bin)
    
    """
    #cv2.imshow("contour", img)
    
    #cv2.imshow("bin", bin)
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()