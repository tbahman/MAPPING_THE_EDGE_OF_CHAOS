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

    mesh = plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap, norm=norm,  alpha=0.8)
    cbar = plt.colorbar(mesh)
    cbar.set_label(metric)

    plt.xlabel('Attention Learning Rate')
    plt.ylabel('FC Learning Rate')
    plt.title(title)
    
    plt.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.8)
    plt.minorticks_on()
    plt.grid(True, which='minor', color='gray', linestyle='--', linewidth=0.4, alpha=0.6)
   
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(10))
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10))

    plt.show()

def calculate_fractal_dimension(Z, fractal_name="Fractal"):

    Z = (Z > 0).astype(int)
    
    p = min(Z.shape)
    n = 2 ** int(floor(np.log2(p)))
    Z = Z[:n, :n]
    
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = []
    
    print(f"Calculating fractal dimension for {fractal_name}...")
