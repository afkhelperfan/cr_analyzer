from re import S
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import json
from matplotlib import pyplot as plt
import pickle
import sys

comp = 1
pos = 1
if len(sys.argv) > 2:
    comp = int(sys.argv[1])
    pos = int(sys.argv[2])


feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224, 224, 3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
    feature_extractor_layer,
])


def convertCVImage(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img


#char_image = json.load(open("char_images/char_image.json"))
char_list = {}

char_temp_mask = json.load(open("char_images/image_temp_mask.json"))
mask_data = char_temp_mask[0]["annotations"]

ss_char_mask = "char_data_4k.json"
ss_mask = json.load(open(ss_char_mask))
ss_data = ss_mask[0]["annotations"]


x = int(mask_data[0]["coordinates"]["x"])
y = int(mask_data[0]["coordinates"]["y"])
w = int(mask_data[0]["coordinates"]["width"]/2)
h = int(mask_data[0]["coordinates"]["height"]/2)




x_ss = int(ss_data[pos-1]["coordinates"]["x"])
y_ss = int(ss_data[pos-1]["coordinates"]["y"])
w_ss = int(ss_data[pos-1]["coordinates"]["width"]/2)
h_ss = int(ss_data[pos-1]["coordinates"]["height"]/2)

input = cv2.imread("data/1/{0}.png".format(comp))
input_resized = cv2.resize(input, (720, 1280))
cv2.imshow("screenshot", input_resized)
cv2.waitKey(1000)

masked_input = input[y_ss-h_ss:y_ss+h_ss, x_ss-w_ss:x_ss+w_ss]
cv2.imshow("input", masked_input)
cv2.waitKey(1000)
masked_input = convertCVImage(masked_input)
i_v = model.predict(preprocess_input(masked_input[None]))


"""
for k, v in char_image.items():
    path = "char_images/{0}_dead.png".format(k)
    print(path)
    char_list[k] = cv2.imread(path)
    char_list[k] = char_list[k][y-h:y+h, x-w:x+w]
    char_list[k] = convertCVImage(char_list[k])
    #cv2.imshow("hello", char_list[k])
"""

# cv2.waitKey(500)

#oden_input = cv2.imread("oden_temp.png")
#oden_input = convertCVImage(oden_input)
#oden_i_v = model.predict(preprocess_input(oden_input[None]))


f = open("character_vector.pickle", "rb")
pred_list = pickle.load(f)

#pred_list = {}
cos_sim_list = {}
for k, v in pred_list.items():
    #pred_list[k] = model.predict(preprocess_input(v[None]))
    cos_sim_list[k] = cosine_similarity(i_v, pred_list[k])


#f = open("character_vector.pickle", "wb+")
# pickle.dump(pred_list,f)

#cos_sim_list = sorted(cos_sim_list.items(), key=lambda x: x[0])

sim_name = list(cos_sim_list.keys())
sim_v = list(cos_sim_list.values())

sim_v = [v[0][0] for v in sim_v]
print(sim_v)
max_kv = max(cos_sim_list.items(), key=lambda x: x[1])

cos_sim_list = dict(sorted(cos_sim_list.items(), key=lambda x: x[1]))
print(max_kv)
sim_name = list(cos_sim_list.keys())
sim_name = sim_name[len(sim_name)-10:len(sim_name)]
sim_damage = [v[0][0] for v in cos_sim_list.values()]
sim_damage = sim_damage[len(sim_damage)-10:len(sim_damage)]
print(sim_damage)
plt.bar(range(len(sim_name)), sim_damage, tick_label=sim_name)

char_image = json.load(open("char_images/char_image.json"))

path = "char_images/{0}".format(char_image[max_kv[0]])
print(path)
char_result = cv2.imread(path)

cv2.imshow("result", char_result)
cv2.waitKey(1000)
plt.title("char match")
plt.xlabel("character")
plt.ylabel("similarity")
plt.show()

cv2.destroyAllWindows()

#oden = cv2.imread("oden_dead.png")
#oden = convertCVImage(oden)
#estrilda = cv2.imread("estrilda_dead.png")
#estrilda = convertCVImage(estrilda)

#oden_v = model.predict(preprocess_input(oden[None]))
#estrilda_v = model.predict(preprocess_input(estrilda[None]))

#cos_sim_o = cosine_similarity(oden_i_v, oden_v)
#cos_sim_e = cosine_similarity(oden_i_v, estrilda_v)

#print("oden : {0}, estrilda : {1}".format(cos_sim_o, cos_sim_e))
