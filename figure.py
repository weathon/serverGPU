from scipy.stats import binomtest
import numpy as np
res0 = np.array()
top1 = np.array(res0[:,:,0])
topn = np.array(res0[:,:,1])
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


import seaborn as sns
import pylab
pylab.rcParams["figure.figsize"] = (6,5)
CI = np.array([[(i/213)-binomtest(i, 213).proportion_ci(0.95).low for i in res] for res in top1])

res = top1/213
sns.heatmap(res, annot=[[f"{res[i,j].round(2)}±{CI[i,j].round(2)}" for j in range(top1.shape[1])] for i in range(top1.shape[0])], fmt = '', yticklabels=range(1,5), xticklabels=range(1,4))
pylab.xlabel("Number of Images")
pylab.title("Top 1 Sequence Level Accuracy")
pylab.ylabel("Number of Beams")
pylab.savefig("top1.png")



res = topn/213
CI = np.array([[(i/213)-binomtest(i, 213).proportion_ci(0.95).low for i in res] for res in topn])

sns.heatmap(res, annot=[[f"{res[i,j].round(2)}±{CI[i,j].round(2)}" for j in range(top1.shape[1])] for i in range(top1.shape[0])], fmt = '', yticklabels=range(1,5), xticklabels=range(1,4))
pylab.xlabel("Number of Images")
pylab.title("Top n Sequence Level Accuracy")
pylab.ylabel("Number of Beams (n)")
pylab.savefig("topn.png")