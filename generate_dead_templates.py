import cv2
import numpy as np
import sys
import json

def adjust(img, alpha=1.0, beta=0.0):
    dst = alpha * img + beta
    return np.clip(dst, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    #if len(sys.argv) > 1:
    #    print(sys.argv[1])

    char_image = json.load(open("char_images/char_image.json"))
    for k,v in char_image.items():
        path = "char_images/{0}".format(v)
        #print(path)
        img = cv2.imread(path)
        #cv2.imshow("image", img)
        #cv2.waitKey(100)
        dst = adjust(img, alpha=0.5)
        #cv2.imshow("dead", dst)
        #cv2.waitKey(100)
        name = "char_images/{0}_dead.png".format(v.replace(".webp", ""))
        print(name)
        cv2.imwrite(name, dst)

    #img = cv2.imread(sys.argv[1])
    #dst = adjust(img, alpha=0.5)
    cv2.destroyAllWindows()
