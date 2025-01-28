
#%%

import matplotlib.pyplot as plt
import glob
import os
import json
import numpy as np

#%%
try: os.chdir("src")
except: pass

#extra = "-extra"
#extra = ""

figure_spec = [
    (f"ucsst-3x2-s*", "dimension=3x2"),
    (f"ucsst-3x3-s*", "dimension=3x3"),
]

all_cpu_list = []
for i,(glob_expr, name) in enumerate(figure_spec):
    dir_list = glob.glob(glob_expr)
    data_list = []
    name_list = []
    for dir_name in dir_list:
        with open(dir_name+"/solving-info.json") as f:
            solving_info = json.load(f)
        cpu = solving_info["time_elapsed"]
        #cpu = solving_info["cpu_times"][0]
        data_list.append(cpu)
        name_list.append(dir_name)
    plt.bar(np.arange(len(data_list))+i*0.2, data_list, width=0.2, alpha=0.3,
            label=name)
    #plt.xticks(np.arange(len(data_list)), name_list
    #           , rotation="vertical")
    plt.xticks(np.arange(len(data_list)), (np.arange(len(data_list))))

    print(glob_expr, np.mean(data_list), np.median(data_list))

plt.ylabel("Running time (seconds)")
plt.xlabel("Simulation index (seed)")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("fig-elapsed-time.pdf", bbox_inches='tight')
plt.savefig("fig-elapsed-time.png", bbox_inches='tight')
plt.show()
