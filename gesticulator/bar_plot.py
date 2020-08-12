import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
x_ticks = ["           speech processing", "             text processing", "               motion inference", "        conversion to angles"]
bert_3s_times = [0.0058255958557128906, 0.7724369859695435, 0.0338973593711853, 0.19182283401489258]
bert_3s_times_std = [0.0017160984235578332, 0.030262767964977733, 0.0008650620483797221, 0.0020550820679680407]
fastT_3s_times = [0.37011232137680056, 0.0335721230506897, 0.026037654876708984, 0.18087475538253783]
fastT_3s_times_std = [0.0016226640305834596, 0.020880673962607905, 0.0004680460116481385, 0.0022869142246917237]
bert_10s_times = [0.01941655158996582, 0.7830116724967957, 0.19037732124328613, 0.5963449120521546]
bert_10s_times_std = [0.007863286977111032, 0.022956405657354323, 0.00631792587263566, 0.006017537381137384]
fastT_10s_times = [0.5042561149597168, 0.12763065099716187, 0.15824578285217286, 0.5860132431983948]
fastT_10s_times_std = [0.0034415932458009453, 0.04256939681034338, 0.0008969402602221561, 0.02191183685661675]
#["BERT+Spectr. (3s input)", "FastText + Prosody (3s input)","BERT+Spectr. (10s input)","FastText + Prosody (3s input)"]
width =0.1
step = 0.1
x_vals = np.arange(len(bert_3s_times))
plt.xticks(x_vals, x_ticks, rotation=20)
plt.bar(x_vals, bert_3s_times, yerr = bert_3s_times_std, width=width, label="BERT+Spectr. (3s input)")
plt.bar(x_vals+ step, fastT_3s_times, yerr= fastT_3s_times_std,  width=width, label="FastText + Prosody (3s input)")
plt.bar(x_vals+ step*2, bert_10s_times, yerr=bert_10s_times_std, width=width, label="BERT+Spectr. (10s input)")
plt.bar(x_vals+ step*3, fastT_10s_times, yerr=fastT_10s_times_std, width=width, label="FastText + Prosody  (10s input)")
plt.legend(loc="upper right")
#plt.show()
#plt.savefig("Next.png")
figure = plt.gcf() # get current figure
figure.set_size_inches(14, 12)
# when saving, specify the DPI
plt.savefig("myplot.png", dpi = 100)