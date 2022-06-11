import cv2
import numpy as np
import sys

def buildPyramidUp(image, maxleval):
    """Build image pyramid for level [0,...,maxlevel]
    """
    imgpyr = [image]
    aux = image
    for i in range(0,maxleval):
        aux = cv2.pyrUp(aux)
        imgpyr.append(aux)

    #for i in range(0, maxleval):
    #    aux = cv2.pyrDown(aux)
    #    imgpyr.append(aux)

    imgpyr.reverse()
    return imgpyr

def buildPyramidDown(image, maxleval):
    """Build image pyramid for level [0,...,maxlevel]
    """
    imgpyr = [image]
    aux = image
    for i in range(0,maxleval):
        aux = cv2.pyrDown(aux)
        imgpyr.append(aux)

    #for i in range(0, maxleval):
    #    aux = cv2.pyrDown(aux)
    #    imgpyr.append(aux)

    imgpyr.reverse()
    return imgpyr

if __name__ == "__main__":

    src = np.array([])

    if len(sys.argv) > 1:
        src = cv2.imread(sys.argv[1])

    oden_template = cv2.imread("oden_dead_cropped.png")
    oden_template_pyr_up = buildPyramidUp(oden_template, 2)
    oden_template_pyr_down = buildPyramidUp(oden_template, 2)
    estrilda_template = cv2.imread("estrilda_dead.png")
    estrilda_template_pyr_up = buildPyramidUp(estrilda_template, 2)
    estrilda_template_pyr_down = buildPyramidDown(estrilda_template, 2)

    max_val_sum = 0
    char_name = ""
    top_left = [0, 0]
    w = 0
    h = 0

    for template in oden_template_pyr_up:
        result = cv2.matchTemplate(src, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print("min_val : {0}, max_val : {1}, min_loc : {2}, max_loc :  {3}".format(min_val, max_val, min_loc, max_loc))
        w, h, _ = template.shape
        top_left = max_loc
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        rect = src.copy()
        print("rect : {0} , {1}".format(top_left, bottom_right))
        cv2.rectangle(rect, top_left, bottom_right, (255, 255, 0), 2)
        cv2.imshow("test", rect)
        cv2.waitKey(1000)
        if max_val > max_val_sum:
            max_val_sum = max_val
            char_name = "oden"

    for template in oden_template_pyr_down:
        result = cv2.matchTemplate(src, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print("min_val : {0}, max_val : {1}, min_loc : {2}, max_loc :  {3}".format(min_val, max_val, min_loc, max_loc))
        w, h, _ = template.shape
        top_left = max_loc
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        rect = src.copy()
        print("rect : {0} , {1}".format(top_left, bottom_right))
        cv2.rectangle(rect, top_left, bottom_right, (255, 255, 0), 2)
        cv2.imshow("test", rect)
        cv2.waitKey(1000)
        if max_val > max_val_sum:
            max_val_sum = max_val
            char_name = "oden"


    """
    for template in estrilda_template_pyr_up:
        result = cv2.matchTemplate(src, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print("min_val : {0}, max_val : {1}, min_loc : {2}, max_loc :  {3}".format(min_val, max_val, min_loc, max_loc))
        w, h, _ = template.shape
        top_left = max_loc
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        rect = src.copy()
        print("rect : {0} , {1}".format(top_left, bottom_right))
        cv2.rectangle(rect, top_left, bottom_right, (255, 255, 0), 2)
        cv2.imshow("test", rect)
        cv2.waitKey(1000)
        if max_val > max_val_sum:
            max_val_sum = max_val
            char_name = "estrilda"


    for template in estrilda_template_pyr_down:
        result = cv2.matchTemplate(src, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print("min_val : {0}, max_val : {1}, min_loc : {2}, max_loc :  {3}".format(min_val, max_val, min_loc, max_loc))
        w, h, _ = template.shape
        top_left = max_loc
        bottom_right = (max_loc[0] + w, max_loc[1] + h)
        rect = src.copy()
        print("rect : {0} , {1}".format(top_left, bottom_right))
        cv2.rectangle(rect, top_left, bottom_right, (255, 255, 0), 2)
        cv2.imshow("test", rect)
        cv2.waitKey(1000)
        if max_val > max_val_sum:
            max_val_sum = max_val
            char_name = "estrilda"
    """
    
    print(char_name)
    print(max_val_sum)

    cv2.destroyAllWindows()

