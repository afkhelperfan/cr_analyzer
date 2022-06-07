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




class OCR:
    def __init__(self, trial, comp = 1, isViz=False, mask_data_path = "dmg_data_4k.json", db_path="data/char_data.db", results_path="data/cr_results.db"):
        self.isViz = isViz
        self.trial = trial
        self.comp  = comp

        #load mask location of the dps digits
        self.mask_data_file = open(mask_data_path)
        self.mask_data = json.load(self.mask_data_file)
        #load char data db
        self.con = sqlite3.connect(db_path)
        self.df = pd.read_sql_query("SELECT * FROM char_data", self.con)
        self.dst_con = sqlite3.connect(results_path)
        #load image and character label of the comp
        self.path = "data/{0}/{1}.png".format(self.trial, self.comp)
        self.label_path = "data/char_label/{0}.json".format(self.comp)
        self.label_data_char = json.load(open(self.label_path))
        self.tree_path = "data/{0}/tree.json".format(self.trial)
        self.tree = json.load(open(self.tree_path))
        self.label_data = self.mask_data[0]["annotations"]

    def load_image(self):
        img = cv2.imread(self.path)
        if self.isViz:
            cv2.imshow("input image", img)
            cv2.waitKey(1000)
        return img

    def set_trial(self, trial):
        self.trial = trial
        self.path = "data/{0}/{1}.png".format(self.trial, self.comp)
        self.label_path = "data/char_label/{0}.json".format(self.comp)
        self.label_data_char = json.load(open(self.label_path))        

    def set_comp(self, comp):
        self.comp = comp
        self.path = "data/{0}/{1}.png".format(self.trial, self.comp)
        self.label_path = "data/char_label/{0}.json".format(self.comp)
        self.label_data_char = json.load(open(self.label_path))

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
    
            masked_img_arr.append(masked_img)
            label_arr.append(label)
            if self.isViz:
                cv2.imshow("mask {0}".format(i), masked_img_arr[i])
                cv2.waitKey(100)

        return masked_img_arr, label_arr

    def segmentate_digits(self, img, thres = 189, kernel=(4,4), isNoDilate = False, counter = 0):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape

        img_amp = ""

        if isNoDilate:
            print("no dilation")
            img_amp = img_gray
        else:
            print("doing dilation {0}".format(counter))
            img_amp = cv2.dilate(img_gray, np.ones(kernel, np.uint8) ,iterations = 1 )


        ret, bin = cv2.threshold(img_amp,thres,255,cv2.THRESH_BINARY)
    


        if self.isViz:
            cv2.imshow("segmentation {0}".format(counter),  bin)
            cv2.waitKey(1000)

        return bin

    def do_ocr(self, img):
        text = pytesseract.image_to_string(img, lang="jpn")
        return self.filter_numbers(text)


    

    def text2score(self, val):
        b_digit = 0
        m_digit = 0

        if val[-1] == "B":
            b_digit = float(val[0:len(val)-1])
            
        elif val[0] == "0":
            pass
        else:
            m_digit = float(val[0:len(val)-1])

        return b_digit, m_digit


    def calculate_scores(self, text_results):
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

    def calculate_b_score(self, text_results):
        for k, v in text_results.items():
            b_digit, m_digit = self.text2score(v)
            text_results[k] = b_digit + m_digit/1000

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

            if v > 5:
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

        #return total_dmg, role_results, char_results

    def record_results(self, total_dmg, role_results, char_results):
        result = {"total_dmg" : total_dmg, "role_results" : role_results, "char_results ": char_results}
        json_data = json.dumps(result)
        output = open("data/{0}/{1}_result.json".format(trial, self.comp), "w+")
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
    if len(sys.argv) > 1:
        trial = int(sys.argv[1])
        isViz = True if len(sys.argv) > 2 else False
        ocr = OCR(trial, isViz=isViz)

        #loop through each comps
        for i in range(1, 7):
            ocr.set_comp(i)
            img = ocr.load_image()
            masked_img_arr, label_arr = ocr.extract_regions(img)
            seg_img_arr = [] 
            seg_img_arr_no = []

            #loop through each dps
            for j in range(len(masked_img_arr)):
                seg_img_arr.append(ocr.segmentate_digits(masked_img_arr[j], counter=j))
                seg_img_arr_no.append(ocr.segmentate_digits(masked_img_arr[j], isNoDilate=True, counter=j))


            text_results = {}
            text_results_no = {}

            for j in range(len(seg_img_arr)):
                text_results[label_arr[j]] = ocr.do_ocr(seg_img_arr[j])
                text_results_no[label_arr[j]] = ocr.do_ocr(seg_img_arr_no[j])

            # calculate score in string(M/B unit)
            text_results = ocr.calculate_scores(text_results)
            text_results_no = ocr.calculate_scores(text_results_no)

            # change numeric units in B
            b_results = ocr.calculate_b_score(text_results)
            b_results_no = ocr.calculate_b_score(text_results_no)


            # calculate misc data
            char_results = ocr.calculate_char_score(b_results)
            role_results = ocr.calculate_role_score(b_results)
            total_dmg = ocr.calculate_sum_score(b_results)
            score = ocr.calculate_confidence_score(b_results)


            # calculate misc data with no dilation
            char_results_no = ocr.calculate_char_score(b_results_no)
            role_results_no = ocr.calculate_role_score(b_results_no)
            total_dmg_no = ocr.calculate_sum_score(b_results_no)
            score_no = ocr.calculate_confidence_score(b_results_no)

            
            print("dilation score : {0}, no dilation score :{1}".format(score, score_no))

            # choose the recognition results which have the higher score
            if score > score_no:
                if isViz:
                    ocr.visualize("result", total_dmg, role_results, char_results)
                ocr.record_results(b_results, total_dmg, role_results, char_results)
            else:
                if isViz:
                    ocr.visualize("result", total_dmg_no, role_results_no, char_results_no)
                ocr.record_results(b_results_no, total_dmg_no, role_results_no, char_results_no)

    else:
        print("Usage : python3 analyze_cr.py [trial]")

