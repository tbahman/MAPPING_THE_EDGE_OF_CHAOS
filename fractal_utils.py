import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from math import log, floor
import numpy as np
import cv2
from PIL import Image


def plot_heatmap(df, s_x, s_y, metric='convergence_measure', title='Convergence Measure Heatmap', cmap=None):
  
    pivot_table = df.pivot(index='fc_lr', columns='transformer_lr', values=metric)
    pivot_table = pivot_table.sort_index().sort_index(axis=1)

    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    norm = mcolors.TwoSlopeNorm(vmin=np.nanmin(Z), vcenter=0, vmax=np.nanmax(Z))
 
    plt.figure(figsize=(s_x, s_y))
    if cmap == None:
        #cmap = plt.get_cmap('RdYlGn')  
        cmap = plt.get_cmap('RdYlBu')  
        cmap = plt.get_cmap('bwr')
        cmap = cmap.reversed()
