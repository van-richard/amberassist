#!/usr/bin/env python
# coding: utf-8

# # MBAR 
# 
# Updated: 12/21/2024 
# 
# By: van

# In[1]:


get_ipython().system('printf "This notebook is found in...\\n$(echo $(realpath .))"')


# In[2]:


import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
# plt.style.use('/home/van/Scripts/bin/v.mplstyle')
import seaborn as sb
from glob import glob
from sklearn.utils import resample

import pymbar
from pymbar.mbar_pmf import mbar_pmf


# In[3]:


os.makedirs('img', exist_ok=True)


# In[4]:


step="step5"
rep="00"

n_windows = 42
val_min = -1.10
val_max = 3.00
fc = 300.0
nbins = n_windows - 1


# In[5]:


val0_k = np.linspace(val_min, val_max, n_windows)
K_k = np.ones(n_windows) * fc


# In[6]:


val_kn = []
for i in range(n_windows):
    fnames = sorted(glob(f'../{i:02d}/{step}.{rep}_equilibration.cv'))
    arrays = [np.loadtxt(f, usecols=1)[::] for f in fnames[:]]
    val_kn.append(np.concatenate(arrays))


# In[7]:


for i in range(n_windows):
    print("Window %02d:" % i, pymbar.timeseries.subsampleCorrelatedData(val_kn[i], conservative=True))
    


# In[8]:


# mbar = mbar_B(val_kn, val0_k, K_k, 300.0, u_kn=np.array(ene_pm3))
mbar = mbar_pmf(val_kn, val0_k, K_k, fc)


# In[9]:


bin_centers, f_i, df_i, reweighting_entropy = mbar.get_pmf(val_min, val_max, nbins)
bin_centers, f_i, df_i, reweighting_entropy = mbar.get_pmf(val_min, val_max, nbins, uncertainties='from-specified', pmf_reference=f_i[:20].argmin())
np.savetxt(f"freefile_mbar_{step}.{rep}", np.column_stack((bin_centers, f_i, df_i)))


# # Histogram + Preliminary PMF 
# 
# ## _Not for publication_

# In[10]:


initial = np.loadtxt(f"freefile_mbar_{step}.{rep}")

xdata=initial[:,0]
ydata=initial[:,1] - initial[:10,1].min()
edata=initial[:,2]

dgd = round(initial[:,1].max() - initial[:10,1].min(),1) # Delta G daggerV
err = round(initial[initial[:,1].argmax()][2], 1) # mbar error

c=sb.color_palette('deep', n_windows)
_xfrac=0.05
_yfrac=0.9

opa=0.4


# In[41]:


# fig = plt.figure(sharex=True, figsize=(7.5,5), dpi=150)
fig = plt.figure(figsize=(10,6))

axs = fig.subplot_mosaic(
    """
    A
    B
    """,height_ratios=[1,2],sharex=True)

for i in range(n_windows):
    sb.kdeplot(val_kn[i], fill=True, alpha=opa, ax=axs['A'], color=c[i])
    axs['A'].axvline(x=val0_k[i], linestyle='--', alpha=opa, color=c[i])
    axs['A'].yaxis.get_major_ticks()[0].label1.set_visible(False)
    axs['A'].grid(linestyle='-', alpha=opa-0.2)

    
    axs['B'].errorbar(xdata, ydata, yerr=edata, linewidth=1, c='black', alpha=opa+0.1)
    axs['B'].scatter(xdata[i-1], ydata[i-1], color=c[i])
    axs['B'].axvline(x=val0_k[i], linestyle='--', alpha=opa, color=c[i])
    axs['B'].grid(linestyle='-', alpha=opa-0.2)
    

axs['B'].annotate(f"Path: {os.getcwd()}", 
                    xy=(_xfrac, _yfrac), xycoords='axes fraction', bbox=dict(fc="w", alpha=opa))
axs['B'].annotate(f"$\Delta G^\ddag$ = {dgd} $\pm$ {err}", 
                    xy=(_xfrac,_yfrac-0.15), xycoords='axes fraction',bbox=dict(fc='w', alpha=opa+0.2))

axs['B'].set_xlabel("r1 - r2 (Å)")
axs['B'].set_ylabel("Potential of Mean Force (kcal/mol)")

plt.margins(x=0.00, y=0.1)
plt.xticks(ticks=val0_k, rotation=55, ha='right')

sb.despine(left=True, bottom=False, right=True, ax=axs['A'])
fig.subplots_adjust(wspace=0, hspace=0)


plt.savefig(f"img/prelim-B-{step}.{rep}.png")
plt.show()


# In[ ]:


# Copy notebook to templates directory

# !cp mbar.ipynb /home/van/Scripts/amber/mbar/mbar.ipynb


# In[ ]:





# In[ ]:




