from pdb import line_prefix
import cv2
import json
import numpy as np
import pytesseract
import sys
import math
import re
import sqlite3
import pandas as pd
import time
import matplotlib.pyplot as plt
from linedetect import StatDetector
from compdetect import CompDetection
import platform
import argparse
import os
 

if platform.system() == "Windows":
    path_tesseract = "C:\\Program Files\\Tesseract-OCR"
    if path_tesseract not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + path_tesseract

class OCR:
    def __init__(self, boss, user, trial, comp = 1, lang="jpn", isViz=False, thres=189, line_detect=False, comp_detect=False, full_scan = False, mask_data_path = "dmg_data_4k.json", db_path="data/char_data.db", results_path="data/cr_results.db"):
        self.isViz = isViz
        self.trial = trial
        self.comp  = comp
        self.user = user
        self.boss = boss
        self.lang = lang
        self.line_detect = line_detect
        self.comp_detect = comp_detect
        self.thres = thres
        self.full_scan = full_scan

        self.statDetector = StatDetector() 
        self.compDetector = CompDetection(isViz=isViz)
        self.mask_data = dict()

        self.label_data = None
        #load mask location of  the dps digits
        if not line_detect:
            print("loading mask data")
            self.mask_data_file = open("data/{0}/{1}/{2}".format(boss, self.user, mask_data_path))
            jsondata = json.load(self.mask_data_file)
            for i in range(1, 7):
                self.mask_data[i] = jsondata
        #load char data db
        self.con = sqlite3.connect(db_path)
        self.df = pd.read_sql_query("SELECT * FROM char_data", self.con)
        self.dst_con = sqlite3.connect(results_path)
        #load image and character label of the comp
        self.path = "data/{0}/{1}/{2}/{3}.png".format(self.boss, self.user, self.trial, self.comp)

        if not comp_detect:
            self.label_path = "data/{0}/{1}/char_label/{2}.json".format(self.boss, self.user, self.comp)
            self.label_data_char = json.load(open(self.label_path))
        self.tree_path = "data/{0}/{1}/{2}/tree.json".format(self.boss, self.user, self.trial)
        self.tree = json.load(open(self.tree_path))
        self.comp_data = dict()


    def init(self):
        self.label_data = self.mask_data[self.comp][0]["annotations"]
        

    def load_image(self):
        print(self.path)
        self.img = cv2.imread(self.path)
        if self.img is None:
            self.img = cv2.imread(self.path.replace("png", "jpg"))
        self.img_h, _, _ = self.img.shape
        if self.isViz:
            cv2.imshow("input image", self.img)
            cv2.waitKey(1000)
        if self.line_detect:
            print("detecting line")
            if not self.comp in self.mask_data.keys():
                self.mask_data[self.comp] = self.statDetector.detect(self.img)
                f = open("data/{0}/{1}/{2}/{3}_dmg_data.json".format(self.boss, self.user, self.trial, self.comp), "w")
                json.dump(self.mask_data[self.comp], f)
        
        return self.img

    def resize_image(self, img):
        if self.img_h != 3840:
            h, w, _ = img.shape
            resized = cv2.resize(img, dsize=(w*5, h*5), interpolation=cv2.INTER_LANCZOS4)
            return resized
        else:
            return img

    def set_trial(self, trial):
        self.trial = trial
        self.path = "data/{0}/{1}/{2}/{3}.png".format(self.boss, self.user, self.trial, self.comp)
        if not self.comp_detect:
            self.label_path = "data/{0}/{1}/char_label/{2}.json".format(self.boss, self.user, self.comp)
            self.label_data_char = json.load(open(self.label_path))
        self.tree_path = "data/{0}/{1}/{2}/tree.json".format(self.boss, self.user, self.trial)
        self.tree = json.load(open(self.tree_path))        

    def set_comp(self):
        if self.comp_detect:
            if not "comp_{0}".format(self.comp) in self.comp_data.keys() or self.full_scan:
                self.comp_data["comp_{0}".format(self.comp)] = self.compDetector.detect(self.img)
                f =  open("data/{0}/{1}/{2}/{3}.json".format(self.boss, self.user, self.trial, self.comp), "w+")
                json.dump(self.comp_data["comp_{0}".format(self.comp)],f)
            self.label_data_char = self.comp_data["comp_{0}".format(self.comp)]
            
        else:
            self.label_path = "data/{0}/{1}/char_label/{2}.json".format(self.boss, self.user, self.comp)
            self.label_data_char = json.load(open(self.label_path))

    def set_round(self, comp):
        self.comp = comp        
        self.path = "data/{0}/{1}/{2}/{3}.png".format(self.boss, self.user, self.trial, self.comp)
        


    def filter_numbers(self, text):
        words = text.split("\n")
        max_score = 0
        max_idx = 0
        print(words)
        for i in range(len(words)):
            score = list(char.isdigit() for char in words[i]).count(True)
            if score >= max_score:
                max_score = score
                max_idx = i

        if max_score == 0:
            return "0"
        else:
            return words[max_idx]

    def extract_regions(self, img):
        masked_img_arr = []
        label_arr = []

        for i in range(len(self.label_data)):
            x = int(self.label_data[i]["coordinates"]["x"])
            y = int(self.label_data[i]["coordinates"]["y"])
            w = int(self.label_data[i]["coordinates"]["width"]/2)
            h = int(self.label_data[i]["coordinates"]["height"]/2)
            label = self.label_data[i]["label"]
            masked_img = img[y-h:y+h, x-w:x+w]
            masked_img = self.resize_image(masked_img)
            masked_img_arr.append(masked_img)
            label_arr.append(label)
            if self.isViz:
                cv2.imshow("mask {0}".format(i), masked_img_arr[i])
                cv2.waitKey(100)
                cv2.imwrite("data/{0}/{1}/{2}/masked_{3}.png".format(self.boss, self.user, self.trial, i+1), masked_img_arr[i])

                
        return masked_img_arr, label_arr

    def segmentate_digits(self, img, kernel=(4,4), isNoDilate = False, counter = 0):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape

        img_amp = ""

        if isNoDilate:
            #print("no dilation")
            img_amp = img_gray
        else:
            #print("doing dilation {0}".format(counter))
            img_amp = cv2.dilate(img_gray, np.ones(kernel, np.uint8) ,iterations = 1 )

        #print("threshold : {0}".format(self.thres))
        ret, bin = cv2.threshold(img_amp,self.thres, 255,cv2.THRESH_BINARY)
        bin = cv2.bitwise_not(bin)


        if self.isViz:
            cv2.imshow("segmentation {0}".format(counter),  bin)
            cv2.imwrite("data/{0}/{1}/{2}/{3}_bin.png".format(self.boss, self.user, self.trial, counter), bin)
            cv2.waitKey(10)

        return bin

    def do_ocr(self, img):
        text = pytesseract.image_to_string(img, lang=self.lang)
        return self.filter_numbers(text)


    

    def text2score(self, val):
        b_digit = 0
        m_digit = 0
        k_digit = 0

        if val[-1] == "B":
            b_digit = float(val[0:len(val)-1])
            
        elif val[0] == "0":
            pass
        elif val[-1] == "M":
            m_digit = float(val[0:len(val)-1])

        elif val[-1] == "K":
            k_digit = float(val[0:len(val)-1])
        return b_digit, m_digit, k_digit


    def calculate_scores(self, text_results):

        if self.lang == "eng":
            print("lang : EN")
            return self.calculate_scores_en(text_results)

        print("lang : JP")
        for k, v in text_results.items():
    
            number = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", v)
            isMan =  v.find("万") != -1
    
            digit = float(number[0])
            isM = isMan or (not isMan and 1 > digit/10) or (not isMan and digit/10 > 9)


            digit = round(digit / 100,2) if isMan else (math.floor(digit*100) if isM else round(digit/10,2)) 
            unit =  "M" if isM else "B"
            val = str(digit) + unit

            text_results[k] = val

        return text_results

    def calculate_scores_en(self, text_results):
        for k, v in text_results.items():
    
            number = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", v)
            isMtext =  v.find("M") != -1
            isKtext = v.find("K") != -1
    
            digit = float(number[0])
            #isM = isMan or (not isMtext and 1 > digit/10) or (not isMan and digit/10 > 9)
            #digit = round(digit / 100,2) if isMan else (math.floor(digit*100) if isM else round(digit/10,2)) 
            unit =  "M" if isMtext else  ("K" if isKtext else "B")
            val = str(digit) + unit

            text_results[k] = val

        return text_results        

    def calculate_b_score(self, text_results):
        for k, v in text_results.items():
            b_digit, m_digit, k_digit = self.text2score(v)
            text_results[k] = b_digit + m_digit/1000 + k_digit/1000000

        return text_results

    def calculate_char_score(self, b_results):
        char_key = []

        for k, v in b_results.items():
            idx = self.df.index[self.df["name"] == self.label_data_char[k]]
            name = self.df.at[idx[0], "name"]
            char_key.append(name)


        char_results = dict.fromkeys(char_key, 0)

        for k, v in b_results.items():
            idx = self.df.index[self.df["name"] == self.label_data_char[k]]
            print(self.label_data_char[k])
            name = self.df.at[idx[0], "name"]
            char_results[name] = v
        
        for k, v in char_results.items():
            char_results[k] = round(v, 2)

        return char_results


    def calculate_role_score(self, b_results):
        role_results = {"fort" : 0, "sorc" : 0, "sus" : 0, "cele" : 0, "might" : 0}
        for k, v in b_results.items():

            idx = self.df.index[self.df["name"] == self.label_data_char[k]]
            role = self.df.at[idx[0], "role"]
            role_results[role] += v
        
        for k, v in role_results.items():
            role_results[k] = round(v, 2)

        return role_results



    def calculate_sum_score(self, b_results):
        total_dmg = 0
        for k, v in b_results.items():
            total_dmg += v

        total_dmg = round(total_dmg,2)
        return total_dmg


    def calculate_confidence_score(self, b_results):

        score = 0

        for k, v in b_results.items():

            idx = self.df.index[self.df["name"] == self.label_data_char[k]]
            iscarry = self.df.at[idx[0], "carry"]

            if v > 8:
                score -= 1                

            elif iscarry == 3:
                score += 1 if v >= 1 else 0
            elif iscarry == 2:
                score +=1 if v >= 0.3 else 0
            elif iscarry == 1:
                score += 1 if (v >= 0.3 and v < 1) else 0
            elif iscarry == 0:
                score += 1 if 0.3 > v else 0

        return score


    

    def visualize(self, title, total_dmg, role_results, char_results):
        print("total damage : {0} B".format(total_dmg))
        print("role damage : {0}".format(role_results))
        print("char damage : {0}".format(char_results))

        if self.isViz:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)

            role_name = list(role_results.keys())
            role_damage = list(role_results.values())

            char_name = list(char_results.keys())
            char_damage = list(char_results.values())


            ax1.bar(range(len(role_results)), role_damage, tick_label=role_name)
            ax1.set_title(title)
            ax1.set_xlabel("Role")
            ax1.set_ylabel("Damage [B]")

            ax2.bar(range(len(char_results)), char_damage, tick_label=char_name)
            ax2.set_title(title)
            ax2.set_xlabel("Characters")
            ax2.set_ylabel("Damage [B]")
            plt.show()
        
        cv2.destroyAllWindows()


    def record_results(self, total_dmg, role_results, char_results):
        result = {"total_dmg" : total_dmg, "role_results" : role_results, "char_results ": char_results}
        json_data = json.dumps(result)
        output = open("data/{0}/{1}/{2}_result.json".format(self.boss, self.trial, self.comp), "w+")
        output.write(json_data)

    def record_results(self, b_results, total_dmg, role_results, char_results):
        result = {"total_dmg" : total_dmg, "role_results" : role_results, "char_results ": char_results}
        json_data = json.dumps(result)


        cursor = self.dst_con.cursor()

        sql = ('''
        INSERT INTO cr_results 
            (trial, comp, total_dps, dps_1, dps_2, dps_3, dps_4, dps_5, name_1, name_2, name_3, name_4, name_5, sus, fort, sorc, cele, might, sus_dps, fort_dps, sorc_dps, cele_dps, might_dps)
        VALUES 
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''')

        data = (self.trial, self.comp, total_dmg,        
        b_results["dps_1"], b_results["dps_2"], b_results["dps_3"], b_results["dps_4"], b_results["dps_5"],
        self.label_data_char["dps_1"], self.label_data_char["dps_2"], self.label_data_char["dps_3"], self.label_data_char["dps_4"], self.label_data_char["dps_5"],
        self.tree["sus"], self.tree["fort"], self.tree["sorc"], self.tree["cele"], self.tree["might"],
        role_results["sus"], role_results["fort"], role_results["sorc"], role_results["cele"], role_results["might"]
        
        )
        print(data)
        
        cursor.execute(sql, data)
        self.dst_con.commit()
        cursor.close()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("boss", help="boss name", type=str)
    parser.add_argument("user", help="username", type=str)
    parser.add_argument("trial_start",help="trial", type=int)
    parser.add_argument("trial_end",help="trial", type=int)
    parser.add_argument("lang", help="language(eng/jpn)", default="eng", type=str)
    parser.add_argument("--bin_thres", help="binarization threshold", default=200, type=int)
    parser.add_argument("--line_detect", help="use automatic stat detection", default=False, action='store_true')
    parser.add_argument("--comp_detect", help="use automatic comp detection", default=False, action='store_true')
    parser.add_argument("--is_viz", help="enable visualization", default=False, action='store_true')
    parser.add_argument("--full_scan", help="do comp scanning and line detection on all trial", default="False", action="store_true")

    args = vars(parser.parse_args())

    boss = args["boss"]
    user = args["user"]
    trial_start = args["trial_start"]
    trial_end = args["trial_end"]
    lang = args["lang"]
    binThres = args["bin_thres"]
    line_detect = args["line_detect"]
    comp_detect = args["comp_detect"]
    isViz= args["is_viz"]
    full_scan = args["full_scan"]
    print(line_detect)
    print(comp_detect)
    print(isViz)
    ocr = OCR(boss, user, trial_start, lang=lang, isViz=isViz, thres=binThres, line_detect= line_detect, comp_detect = comp_detect, full_scan = full_scan, mask_data_path="dmg_data.json", results_path="data/{0}/{1}/cr_results.db".format(boss, user))

    for k in range(trial_start, trial_end + 1):
        ocr.set_trial(k)

        for i in range(1, 7):       
            ocr.set_round(i)
            img = ocr.load_image()
            ocr.set_comp()
            ocr.init()
            masked_img_arr, label_arr = ocr.extract_regions(img)
            seg_img_arr = [] 

            #loop through each dps
            for j in range(len(masked_img_arr)):
                seg_img_arr.append(ocr.segmentate_digits(masked_img_arr[j], counter=5*(i-1)+(j+1), isNoDilate=False))


            text_results = {}

            for j in range(len(seg_img_arr)):
                text_results[label_arr[j]] = ocr.do_ocr(seg_img_arr[j])

            # calculate score in string(M/B unit)
            text_results = ocr.calculate_scores(text_results)

            # change numeric units in B
            b_results = ocr.calculate_b_score(text_results)


            # calculate misc data
            char_results = ocr.calculate_char_score(b_results)
            role_results = ocr.calculate_role_score(b_results)
            total_dmg = ocr.calculate_sum_score(b_results)

            ocr.record_results(b_results, total_dmg, role_results, char_results)
            cv2.destroyAllWindows()

    """
    if len(sys.argv) > 2:
        boss = sys.argv[1]
        user = sys.argv[2]
        trial = int(sys.argv[3])
        lang = sys.argv[4] if len(sys.argv) > 4 else "eng"
        binThres = int(sys.argv[5]) if len(sys.argv) > 5 else 189
        isViz = True if len(sys.argv) > 6 else False
        ocr = OCR(boss, user, trial, lang=lang, isViz=isViz, thres=binThres, mask_data_path="dmg_data.json", results_path="data/{0}/{1}/cr_results.db".format(boss, user))

        #loop through each comps
        for i in range(1, 7):
            ocr.set_comp(i)
            img = ocr.load_image()
            masked_img_arr, label_arr = ocr.extract_regions(img)
            seg_img_arr = [] 

            #loop through each dps
            for j in range(len(masked_img_arr)):
                seg_img_arr.append(ocr.segmentate_digits(masked_img_arr[j], counter=5*(i-1)+(j+1), isNoDilate=False))


            text_results = {}

            for j in range(len(seg_img_arr)):
                text_results[label_arr[j]] = ocr.do_ocr(seg_img_arr[j])

            # calculate score in string(M/B unit)
            text_results = ocr.calculate_scores(text_results)

            # change numeric units in B
            b_results = ocr.calculate_b_score(text_results)


            # calculate misc data
            char_results = ocr.calculate_char_score(b_results)
            role_results = ocr.calculate_role_score(b_results)
            total_dmg = ocr.calculate_sum_score(b_results)

            ocr.record_results(b_results, total_dmg, role_results, char_results)

    else:
        print("Usage : python3 analyze_cr.py [boss] [user] [trial] [lang] [vis]")

    """