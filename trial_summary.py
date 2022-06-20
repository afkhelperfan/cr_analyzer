from codecs import ignore_errors
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import sqlite3
import pandas as pd
import math
import json
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

plt.switch_backend('QT4Agg') #default on my system
print('#3 Backend:',plt.get_backend())

if __name__ == "__main__":
    con = sqlite3.connect("data/cr_results.db")
    df = pd.read_sql_query("SELECT * FROM cr_results", con)

    df.to_excel("reports/summary.xlsx", index=False)
    char_dps = {}
    char_over_1b = {}
    tree = {}

    tree_set_df = pd.DataFrame(df.loc[:, "sus":"might"]).drop_duplicates()
    #print(tree_set_df.columns.values.tolist())
    tree_damage_record = pd.DataFrame()


    #print(char_df.columns)

    char_dps = {}

    tree_wise_df = []
    #tree_merge_df = pd.DataFrame(columns=df.columns.values.tolist())

    for index, rows in tree_set_df.iterrows():
        tree_individual = df[df["sus"] == rows["sus"]][df["fort"] == rows["fort"]][df["cele"] == rows["cele"]][df["sorc"] == rows["sorc"]][df["might"] == rows["might"]]
        tree_wise_df.append(tree_individual)

    #print(len(tree_wise_df))
    for tree_wise in tree_wise_df:

        
        char_df = pd.DataFrame(tree_wise.loc[:, "dps_1":"name_5"])
        for index, rows in char_df.iterrows():
            for i in range(5):
                name = rows["name_{0}".format(i+1)]
                dps = rows["dps_{0}".format(i+1)]
                if not name in char_dps:
                    char_dps[name] = np.array([])
                char_dps[name] = np.append(char_dps[name] ,dps)

        char_name = char_dps.keys()
        char_dps_v = char_dps.values()

        #for k, v in char_dps.items():
        #    print("{0} : avg {1} B, min {2} B, max {3} B, std +/- {4} B".format(k, round(v.mean(),2), v.min(), v.max(), round(v.std(),2)))


        comp_dmg = []
        tree_total_dmg = pd.DataFrame()
        tree_avg_dmg = pd.DataFrame()
        for i in range(1,7):
            #print("comp {0} statistics".format(i))
            #print("average damage : {0} B".format(round(tree_wise[tree_wise["comp"] == i].loc[:, "total_dps"].mean(),2)))
            comp_dmg.append(tree_wise[tree_wise["comp"] == i].loc[:, "total_dps"].values.tolist())
            #print("stdev : +/- {0} B".format(round(tree_wise[tree_wise["comp"] == i].loc[:, "total_dps"].std(),2)))
            #print("max : {0} B".format(tree_wise[tree_wise["comp"] == i].loc[:, "total_dps"].max()))
            #print("min : {0} B".format(tree_wise[tree_wise["comp"] == i].loc[:, "total_dps"].min()))
            tree_dmg = round(tree_wise[tree_wise["comp"] == i].loc[:, "sus_dps":"might_dps"].max(),2)

            tree_avg  =round(tree_wise[tree_wise["comp"] == i].loc[:, "sus_dps":"might_dps"].mean(),2)
            #tree_dmg = tree_wise[tree_wise["comp"] == i].loc[:, "sus_dps":"might_dps"]
            #tree_dmg_max = round(tree_wise[tree_wise["comp"] == i].loc[:, "sus_dps":"might_dps"].max(),2)
            tree_total_dmg = tree_total_dmg.append(tree_dmg,ignore_index=True)
            tree_avg_dmg = tree_avg_dmg.append(tree_avg, ignore_index=True)
            #tree_total_dmg.append(tree_dmg)
    

        colormap = {"sorc" : "purple", "might" : "red", "cele" : "green", "sus": "blue", "fort": "yellow"}

        #print(tree_total_dmg)
        fig = plt.figure(figsize=(16, 9))

        #tree_sum_dmg = pd.DataFrame(columns=tree_total_dmg.columns.tolist())
        #print(tree_total_dmg)
        #for index, column in  tree_total_dmg.iterrows():
        #    if index == 5:
        #        break

        #    role_sum = tree_total_dmg.iloc[index] + tree_total_dmg.iloc[index + 10] + tree_total_dmg.iloc[index + 15] + tree_total_dmg.iloc[index + 20] + tree_total_dmg.iloc[index+25]
        #    tree_sum_dmg = tree_sum_dmg.append(role_sum, ignore_index=True)

        
        #print(tree_sum_dmg)

        #tree = round(tree_wise[tree_wise["comp"] == 0].loc[:, "sus_dps":"might_dps"].mean())
        tree = tree_wise.loc[:, "sus":"might"].mean()


        plt.subplot(2,2,1)
        max_s = tree_total_dmg.sum()
        max_s_p = max_s.rename(lambda x: x.replace("_dps", ""))
        max_s = max_s.rename(lambda x: x.replace("_dps", "_max"))
        max_s_p.plot.bar(title="role damage",color=[colormap[c]for c in max_s_p.keys()])
        plt.grid()
        plt.xlabel("role")
        plt.yticks([i for i in range(10+1)])
        plt.ylabel("max damage [B]")

        avg = tree_avg_dmg.sum()
        ax1 = plt.subplot(2,2,2)
        avg_p = avg.rename(lambda x: x.replace("_dps", ""))
        avg = avg.rename(lambda x: x.replace("_dps", "_avg"))
        avg_p.plot.bar(title="role damage",color=[colormap[c]for c in avg_p.keys()])
        plt.grid()
        plt.xlabel("role")
        plt.yticks([i for i in range(10+1)])
        plt.ylabel("avg damage [B]")


        max_s = max_s.rename(lambda x: x.replace("_dps", "_max"))
        tree_all = pd.concat([tree, max_s], axis=0)
        #print(tree_all) 

        avg = avg.rename(lambda x: x.replace("_dps", "_avg"))
        tree_all = pd.concat([tree_all, avg], axis=0)
        tree_damage_record = tree_damage_record.append(tree_all, ignore_index=True)


        #print(tree_all)
        
        #tree_damage_record = tree_damage_record.append(max_s, ignore_index=True)
        #print(tree_damage_record)
        #tree_total_dmg.boxplot()





        ax2 = plt.subplot(2,2,3)
    

        #ax2.bar([i+1 for i in range(6)], comp_dmg, yerr=comp_std)

        ax2.boxplot(comp_dmg)
        ax2.set_title("comp damage")
        ax2.set_xlabel("comp")
        ax2.set_ylabel("damage [B]")

        ax3 = plt.subplot(2,2,4)
        tree = tree.rename(lambda x: x.replace("_dps", ""))
        tree.plot.bar(title="tree lvl", color=[colormap[c]for c in tree.keys()])
        plt.xlabel("role")
        plt.ylabel("level")
        plt.ylim([0,165])

        tree_l_list = tree.index.tolist()
        tree_list = tree.values.tolist()
        tree_map = {}
        tree_str = ""
        for i in range(len(tree_list)):
            islast = True if len(tree_list) -i == 1 else False
            tree_map[tree_l_list[i]] = tree_list[i]
            prefix = "{0}-{1}{2}".format(tree_l_list[i], tree_list[i], "-" if islast else "")
            tree_str = tree_str + prefix

        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        #fig.set_dpi(1000)
        fig.savefig("reports/{0}-overview.png".format(tree_str), dpi=500)

        plt.close(fig)

        #plt.show()

        #ax4 = plt.subplot(2,2,4)

        fig = plt.figure(figsize=(16,9))
        box = fig.add_subplot(1,1,1)
        box.boxplot(char_dps_v)
        box.set_title("char damage at {0}".format(str(tree_map)))
        box.set_xlabel("character")
        box.set_ylabel("damage [B]")
        box.set_xticks([i+1 for i in range(len(char_name))])
        box.set_xticklabels(char_name, fontsize=8)
        box.set_ylim([0, 10])
        #figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()
        #fig.set_dpi(1000)

        #fig2 = plt.gcf()
        fig.savefig("reports/{0}-character.png".format(tree_str), dpi=500)
        #plt.show()

        
        fig = plt.figure()
        ax_arr = []
        for i in range(len(comp_dmg)):

            comp_dmg[i] = np.array(comp_dmg[i])
            #kde = gaussian_kde(comp_dmg[i])

            
            ax_arr.append(fig.add_subplot(3,2,i+1))
            ax_arr[i].hist(comp_dmg[i])
            #ax_arr[i].plot(kde(np.linspace(0, 32, num=10)))
            ax_arr[i].set_title("damage distribution of comp {0}".format(i + 1))
            ax_arr[i].set_ylabel("amount")
            ax_arr[i].set_xlabel("damage [B]")
            plt.tight_layout()

        fig.savefig("reports/{0}-damage-distrib.png".format(tree_str), dpi=500)
        plt.close(fig)


    role_arr = ["might", "sorc", "cele", "sus", "fort"]
    
    for role in role_arr:
        ax_arr = []
        fig = plt.figure()
        for i in range(1,7):
            ax_arr.append(fig.add_subplot(3,2,i))
            comp = df[df["comp"] == i]
            lvl = comp.loc[:,role]
            score = comp.loc[:, "dps_1"] + comp.loc[:, "dps_2"] + comp.loc[:, "dps_3"] + comp.loc[:, "dps_4"] + comp.loc[:, "dps_5"]
            ax_arr[i-1].scatter(lvl.values.tolist(), score.values.tolist())
            coef_M = np.corrcoef(lvl.values.tolist(), score.values.tolist())
            r = round(coef_M[0][1],2)
            if r > 0.7 or r < -0.7:
                theta = np.polyfit(lvl.values.tolist(), score.values.tolist(), 1)
                equ_line = theta[1] + theta[0] * np.array(lvl.values.tolist())
                ax_arr[i-1].plot(lvl.values.tolist(), equ_line, color="red")

            ax_arr[i-1].set_title("comp {0} dmg and {1}, r = {2}".format(i , role, r))
            ax_arr[i-1].set_ylabel("dmg [B]")
            ax_arr[i-1].set_xlabel("{0} Lvl".format(role))

        fig.savefig("reports/{0}-corr.png".format(role), dpi=500)
        plt.close(fig)
        #plt.show()

    #ax = fig.add_subplot(111, projection='3d')
    #ax.set_title("Helix", size = 20)


    #might_lvl = df.loc[:,"might"]
    #sorc_lvl = df.loc[:, "sorc"]
    #score = df[df["comp"] == 1].loc[:, "total_dps"] + df[df["comp"] == 2].loc[:, "total_dps"] + df[df["comp"] == 3].loc[:, "total_dps"] + df[df["comp"] == 4].loc[:, "total_dps"] + df[df["comp"] == 5].loc[:, "total_dps"] + df[df["comp"] == 6].loc[:, "total_dps"]
    #print(score)
    #ax.set_xlabel("might damage", size = 14)
    #ax.set_ylabel("sorc damage", size = 14)
    #ax.set_zlabel("total damage", size = 14)
    #ax.scatter(might_lvl.values.tolist(), sorc_lvl.values.tolist(), score.values.tolist())
    #plt.show()
    """
    ax2 = fig.add_subplot(3,2,2)
    comp_1 = df[df["comp"] == 2]
    might_lvl = comp_1.loc[:,"might"]
    might_score = comp_1.loc[:, "dps_1"] + comp_1.loc[:, "dps_2"] + comp_1.loc[:, "dps_3"] + comp_1.loc[:, "dps_4"] + comp_1.loc[:, "dps_5"]
    ax2.scatter(might_lvl.values.tolist(), might_score.values.tolist())
    ax2.set_title("comp 2 dmg and might relationship")
    ax2.set_ylabel("dmg [B]")
    ax2.set_xlabel("Might Lvl")


    ax3 = fig.add_subplot(3,2,3)
    comp_1 = df[df["comp"] == 3]
    might_lvl = comp_1.loc[:,"might"]
    might_score = comp_1.loc[:, "dps_1"] + comp_1.loc[:, "dps_2"] + comp_1.loc[:, "dps_3"] + comp_1.loc[:, "dps_4"] + comp_1.loc[:, "dps_5"]
    ax3.scatter(might_lvl.values.tolist(), might_score.values.tolist())
    ax3.set_title("comp 3 dmg and might relationship")
    ax3.set_ylabel("dmg [B]")
    ax3.set_xlabel("Might Lvl")


    ax4 = fig.add_subplot(3,2,4)
    comp_1 = df[df["comp"] == 4]
    might_lvl = comp_1.loc[:,"might"]
    might_score = comp_1.loc[:, "dps_1"] + comp_1.loc[:, "dps_2"] + comp_1.loc[:, "dps_3"] + comp_1.loc[:, "dps_4"] + comp_1.loc[:, "dps_5"]
    ax4.scatter(might_lvl.values.tolist(), might_score.values.tolist())
    ax4.set_title("comp 4 dmg and might relationship")
    ax4.set_ylabel("dmg [B]")
    ax4.set_xlabel("Might Lvl")



    ax5 = fig.add_subplot(3,2,5)
    comp_1 = df[df["comp"] == 5]
    might_lvl = comp_1.loc[:,"might"]
    might_score = comp_1.loc[:, "dps_1"] + comp_1.loc[:, "dps_2"] + comp_1.loc[:, "dps_3"] + comp_1.loc[:, "dps_4"] + comp_1.loc[:, "dps_5"]
    ax5.scatter(might_lvl.values.tolist(), might_score.values.tolist())
    ax5.set_title("comp 5 dmg and might relationship")
    ax5.set_ylabel("dmg [B]")
    ax5.set_xlabel("Might Lvl")



    ax6 = fig.add_subplot(3,2,6)
    comp_1 = df[df["comp"] == 6]
    might_lvl = comp_1.loc[:,"might"]
    might_score = comp_1.loc[:, "dps_1"] + comp_1.loc[:, "dps_2"] + comp_1.loc[:, "dps_3"] + comp_1.loc[:, "dps_4"] + comp_1.loc[:, "dps_5"]
    ax6.scatter(might_lvl.values.tolist(), might_score.values.tolist())
    ax6.set_title("comp 6 dmg and might relationship")
    ax6.set_ylabel("dmg [B]")
    ax6.set_xlabel("Might Lvl")
    """


    

        
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    might_sorted = tree_damage_record.sort_values("might")
    might_lvl = might_sorted.loc[:, "might"]
    #might_lvl = tree_damage_record.loc[:, "might"]
    might_max = might_sorted.loc[:, "might_max"]
    might_avg = might_sorted.loc[:, "might_avg"]
    might_table_max = pd.concat([might_lvl, might_max], axis=1)
    print(might_table_max)

    x = np.array(might_lvl.values.tolist())
    y_max = np.array(might_max.values.tolist())
    y_avg = np.array(might_avg.values.tolist())


    theta_max = np.polyfit(x, y_max, 1)
    theta_avg = np.polyfit(x,y_avg, 1)
    y_max_l = theta_max[1] + theta_max[0] * x
    y_avg_l = theta_avg[1] + theta_avg[0] * x
    ax1.plot(x, y_max_l, label="max")
    ax1.scatter(x, y_max)
    ax1.plot(x, y_avg_l, label="avg")
    ax1.scatter(x, y_avg)
    ax1.set_title("might tree / dmg")
    ax1.set_ylabel("dmg [B]")
    ax1.set_xlabel("might lvl")
    ax1.set_yticks([i+1 for i in range(11)])
    ax1.set_ylim([0,10])
    


    ax2 = fig.add_subplot(2,2,2)
    sorc_sorted = tree_damage_record.sort_values("sorc")
    sorc_lvl = sorc_sorted.loc[:, "sorc"]
    #might_lvl = tree_damage_record.loc[:, "might"]
    sorc_max = sorc_sorted.loc[:, "sorc_max"]
    sorc_avg = sorc_sorted.loc[:, "sorc_avg"]
    sorc_table_max = pd.concat([sorc_lvl, sorc_max], axis=1)

    x = np.array(sorc_lvl.values.tolist())
    y_max = np.array(sorc_max.values.tolist())
    y_avg = np.array(sorc_avg.values.tolist())


    theta_max = np.polyfit(x, y_max, 1)
    theta_avg = np.polyfit(x,y_avg, 1)
    y_max_l = theta_max[1] + theta_max[0] * x
    y_avg_l = theta_avg[1] + theta_avg[0] * x
    ax2.plot(x, y_max_l, label="max")
    ax2.scatter(x, y_max)
    ax2.plot(x, y_avg_l, label="avg")
    ax2.scatter(x, y_avg)
    ax2.set_title("sorc tree / dmg")
    ax2.set_ylabel("dmg [B]")
    ax2.set_xlabel("sorc lvl")
    ax2.set_yticks([i+1 for i in range(11)])
    ax2.set_ylim([0,10])


    ax3 = fig.add_subplot(2,2,3)
    cele_sorted = tree_damage_record.sort_values("cele")
    cele_lvl = cele_sorted.loc[:, "cele"]
    #might_lvl = tree_damage_record.loc[:, "might"]
    cele_max = cele_sorted.loc[:, "cele_max"]
    cele_avg = cele_sorted.loc[:, "cele_avg"]
    cele_table_max = pd.concat([cele_lvl, cele_max], axis=1)

    x = np.array(cele_lvl.values.tolist())
    y_max = np.array(cele_max.values.tolist())
    y_avg = np.array(cele_avg.values.tolist())


    theta_max = np.polyfit(x, y_max, 1)
    theta_avg = np.polyfit(x,y_avg, 1)
    y_max_l = theta_max[1] + theta_max[0] * x
    y_avg_l = theta_avg[1] + theta_avg[0] * x
    ax3.plot(x, y_max_l, label="max")
    ax3.scatter(x, y_max)
    ax3.plot(x, y_avg_l, label="avg")
    ax3.scatter(x, y_avg)
    ax3.set_title("cele tree / dmg")
    ax3.set_ylabel("dmg [B]")
    ax3.set_xlabel("sorc lvl")
    ax3.set_yticks([i+1 for i in range(11)])
    ax3.set_ylim([0,10])


    
    #plt.show()


    
    plt.show()

    """

















    char_df = pd.DataFrame(df.loc[:, "dps_1":"name_5"])
    for index, rows in char_df.iterrows():
        for i in range(5):
            name = rows["name_{0}".format(i+1)]
            dps = rows["dps_{0}".format(i+1)]
            if not name in char_dps:
                char_dps[name] = np.array([])
            char_dps[name] = np.append(char_dps[name] ,dps)

    char_name = char_dps.keys()
    char_dps_v = char_dps.values()

    for k, v in char_dps.items():
        print("{0} : avg {1} B, min {2} B, max {3} B, std +/- {4} B".format(k, round(v.mean(),2), v.min(), v.max(), round(v.std(),2)))


    comp_dmg = []
    for i in range(1,7):
        print("comp {0} statistics".format(i))
        print("average damage : {0} B".format(round(df[df["comp"] == i].loc[:, "total_dps"].mean(),2)))
        comp_dmg.append(df[df["comp"] == i].loc[:, "total_dps"].values.tolist())
        print("stdev : +/- {0} B".format(round(df[df["comp"] == i].loc[:, "total_dps"].std(),2)))
        print("max : {0} B".format(df[df["comp"] == i].loc[:, "total_dps"].max()))
        print("min : {0} B".format(df[df["comp"] == i].loc[:, "total_dps"].min()))
        tree_dmg = round(df[df["comp"] == i].loc[:, "sus_dps":"might_dps"].mean(),2)
        tree_total_dmg = tree_total_dmg.append(tree_dmg,ignore_index=True)
        #tree_total_dmg.append(tree_dmg)
    

    colormap = {"sorc" : "purple", "might" : "red", "cele" : "green", "sus": "blue", "fort": "yellow"}

    fig = plt.figure()

    tree = round(df[df["comp"] == 0].loc[:, "sus_dps":"might_dps"].mean())

    plt.subplot(2,2,1)
    avg = tree_total_dmg.sum()
    avg = avg.rename(lambda x: x.replace("_dps", ""))
    avg.plot.bar(title="role damage",color=[colormap[c]for c in avg.keys()])

    plt.xlabel("role")
    plt.ylabel("average damage [B]")

    ax2 = plt.subplot(2,2,2)
    
    #ax2.bar([i+1 for i in range(6)], comp_dmg, yerr=comp_std)

    ax2.boxplot(comp_dmg)
    ax2.set_title("comp damage")
    ax2.set_xlabel("comp")
    ax2.set_ylabel("damage [B]")

    ax3 = plt.subplot(2,2,3)
    tree = tree.rename(lambda x: x.replace("_dps", ""))
    tree.plot.bar(title="tree lvl")
    plt.xlabel("role")
    plt.ylabel("level")
    plt.ylim([0,165])

    #plt.show()

    #ax4 = plt.subplot(2,2,4)
    plt.boxplot(char_dps_v)
    plt.title("char damage")
    plt.xlabel("character")
    plt.ylabel("damage [B]")

    plt.xticks([i+1 for i in range(len(char_name))], char_name)
    plt.ylim([0, 10])

    #plt.show()



    """
    
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
