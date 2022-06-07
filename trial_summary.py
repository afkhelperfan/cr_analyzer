import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import sqlite3
import pandas as pd


if __name__ == "__main__":
    con = sqlite3.connect("data/cr_results.db")
    df = pd.read_sql_query("SELECT * FROM cr_results", con)

    char_dps = {}
    char_over_1b = {}
    tree = {}

    for idx, row in df.iterrows():
        char_dps[row["name_1"]] = row["dps_1"]
        char_dps[row["name_2"]] = row["dps_2"]
        char_dps[row["name_3"]] = row["dps_3"]
        char_dps[row["name_4"]] = row["dps_4"]
        char_dps[row["name_5"]] = row["dps_5"]
        tree["sus"] = row["sus"]
        tree["fort"] = row["fort"]
        tree["sorc"] = row["sorc"]
        tree["cele"] = row["cele"]
        tree["might"] = row["might"]
        print(row["sus"])

    char_over_1b = dict(filter(lambda item: item[1] > 1,  char_dps.items()))

    print(char_dps)
    role = ["sus", "fort", "sorc", "cele", "might"]
    c = ["blue", "yellow", "purple", "green", "red"]
    role_dps  = [df["sus_dps"].sum(), df["fort_dps"].sum(), df["sorc_dps"].sum(), df["cele_dps"].sum(), df["might_dps"].sum()]

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    

    ax1.bar(range(5), role_dps, tick_label=role, color=c)
    ax1.set_title("Role Total Damage")
    ax1.set_xlabel("Role")
    ax1.set_ylabel("Damage [B]")


    tree_name = list(tree.keys())
    tree_lvl = list(tree.values())
    
    ax2.bar(range(5), tree_lvl, tick_label=tree_name, color=c)
    ax2.set_title("Tree Lvl")
    ax2.set_xlabel("Role")
    ax2.set_ylabel("Lvl")
    #fig.show()

    char_dps = dict(sorted(char_dps.items(), key=lambda x:x[1]))
    char_name = list(char_dps.keys())
    char_name = char_name[len(char_name)-10:len(char_name)]
    char_damage = list(char_dps.values())
    char_damage = char_damage[len(char_damage)-10:len(char_damage)]

    ax3.bar(range(len(char_name)), char_damage, tick_label=char_name)
    ax3.set_title("Main Carry")
    ax3.set_xlabel("Character")
    ax3.set_ylabel("Damage [B]")
    plt.show()
    

"""
results = []
isViz = False


if len(sys.argv) > 1:
    trial = sys.argv[1]
    paths = ["data/{0}/{1}_result.json".format(trial,i) for i in range(1, 7)]
    if(len(sys.argv) > 2):
        isViz = True   


for i in range(len(paths)):
    json_data = open(paths[i], "r")
    result = json.load(json_data)
    results.append(result)


print(results)

role_summary = {"sus" : 0, "fort" : 0, "sorc" : 0, "might" : 0, "cele" : 0}

char_summary = {}


for i in range(len(results)):
    role_summary["sus"] += results[i]["role_results"]["sus"]
    role_summary["fort"] += results[i]["role_results"]["fort"]
    role_summary["sorc"] += results[i]["role_results"]["sorc"]
    role_summary["might"] += results[i]["role_results"]["might"]
    role_summary["cele"] += results[i]["role_results"]["cele"]
    for k, v in results[i]["char_results "].items():
        char_summary[k] = v



role_name = list(role_summary.keys())
role_damage = list(role_summary.values())

char_name = list(char_summary.keys())
char_damage = list(char_summary.values())

if isViz:
    plt.bar(range(len(role_summary)), role_damage, tick_label=role_name)
    plt.xlabel("Role")
    plt.ylabel("Damage [B]")
    plt.show()

    plt.bar(range(len(char_summary)), char_damage, tick_label=char_name)
    plt.xlabel("Character")
    plt.ylabel("Damage [B]")
    plt.show()

role_total = json.dumps(role_summary)
char_total = json.dumps(char_summary)
role_total_f = open("data/{0}/role_total.json".format(trial), "w+")
role_total_f.write(role_total)
char_total_f = open("data/{0}/char_total.json".format(trial), "w+")
char_total_f.write(char_total)
"""