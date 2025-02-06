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
    for size in sizes:
        
          num_blocks = n // size
          count = 0
          for i in range(num_blocks):
              for j in range(num_blocks):
                  block = Z[i*size:(i+1)*size, j*size:(j+1)*size]
                  if np.any(block):
                      count += 1
          counts.append(count)
          print(f"Box size: {size}, Count: {count}")
  
    sizes = np.array(sizes)
    counts = np.array(counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dimension = -coeffs[0]
  
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(sizes), np.log(counts), 'o-', label='Box-Counting Data')
    plt.plot(np.log(sizes), np.polyval(coeffs, np.log(sizes)), 'r--', 
             label=f'Fit Line (D={fractal_dimension:.4f})')
    plt.xlabel('log(Box size)')
    plt.ylabel('log(Count)')
    plt.title(f'Box-Counting for Fractal Dimension of {fractal_name}')
    plt.legend()
    plt.grid(True)
    plot_filename = f"box_counting_{fractal_name.replace(' ', '_').lower()}.png"
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
    
    print(f"Estimated Fractal Dimension of {fractal_name}: {fractal_dimension:.4f}\n")
    
    return fractal_dimension

def detect_edges(fractal_img, upsample_factor=1, low_threshold=50, high_threshold=150):

    if isinstance(fractal_img, np.ndarray):

        if len(fractal_img.shape) == 3: 
            fractal_img = cv2.cvtColor(fractal_img, cv2.COLOR_BGR2GRAY)
    else:
        fractal_img = fractal_img.convert("L")
        fractal_img = np.array(fractal_img)  

    _, binary_array = cv2.threshold(fractal_img, 127, 255, cv2.THRESH_BINARY)
 
    if upsample_factor > 1:
        height, width = binary_array.shape
        upsampled_array = cv2.resize(
            binary_array, 
            (width * upsample_factor, height * upsample_factor), 
            interpolation=cv2.INTER_NEAREST
        )
    else:
        upsampled_array = binary_array

    grad_x = cv2.Sobel(upsampled_array, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(upsampled_array, cv2.CV_64F, 0, 1, ksize=3)
   
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
  
    _, edges = cv2.threshold(magnitude, high_threshold, 255, cv2.THRESH_BINARY)

    return edges
