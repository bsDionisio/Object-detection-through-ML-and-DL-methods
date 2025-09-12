import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import collections as mplcollections
from matplotlib import colors as mcolors
from torch_dimcheck import dimchecked
from torchtyping import TensorType

from disk import MatchedPairs

#Utility for displaying two images side-by-side or stacked vertically in a single matplotlib figure, with optional grid overlay
class MultiFigure:
    @dimchecked
    def __init__(
        self,
        image1: TensorType['H', 'W', 'C'],
        image2: TensorType['H', 'W', 'C'],
        grid=None,
        vertical=False,
    ):
        #Ensures both images are the same size
        assert image1.shape == image2.shape
        #Extracts height, width and channels for later use
        h, w, c = image1.shape

        #If vertical=True, concatenate along rows (dim=0) -> one image stacked above the other; 
        # if vertical=False, concatenate along columns (dim=1) -> images placed side by side
        cat_dim = 0 if vertical else 1
        images = torch.cat([image1, image2], dim=cat_dim)

        #Tall figure is stacked vertically; Wide figure if placed side by side
        figsize = (20, 40) if vertical else (40, 20)

        #Creates a single subplot figure
        self.fig, self._ax = plt.subplots(
            figsize=figsize,
            frameon=False,
            constrained_layout=True
        )
        #Displays the concatenated image
        self._ax.imshow(images)
        xmax = w
        ymax = h
        #If vertical -> height doubles
        if vertical:
            ymax *= 2
        #If horizontal -> width doubles
        else:
            xmax *= 2

        self._ax.set_xlim(0, xmax)
        self._ax.set_ylim(ymax, 0)  #Flips y-axis to match image coordinates

        #If no grid, axes are hidden
        if grid is None:
            self._ax.axis('off')
        #If grid is given (integer), draws a grid with spacing grid pixels
        else:
            self._ax.set_xticks(np.arange(0, xmax, grid))
            self._ax.set_yticks(np.arange(0, ymax, grid))
            self._ax.grid()

        #If vertical -> second image starts h pixels down
        if vertical:
            self.offset = torch.tensor([0, h])
        #If horizontal -> second image starts w pixels to the right
        else:
            self.offset = torch.tensor([w, 0])

    @dimchecked
    def mark_xy(
        self,
        xy1: TensorType[2, 'N'],    #First row: x-coordinates
        xy2: TensorType[2, 'N'],    #Second row: y-coordinates
        color='green',              #line colour (default "green")
        lines=True,                 #whether to draw connecting lines
        marks=True,                 #whether to draw point markers
        plot_n=None,                #max number of correspondences to plot (for downsampling)
        linewidth=None,             #style controls
        marker_size=None,
    ):
        #Offset adjustment; So each entry is a line segment between two points
        xy2 = xy2 + self.offset.reshape(2, 1)

        #Stack coordinates for plotting
        xys = torch.stack([xy1.T, xy2.T], dim=1)

        #Optional downsampling; If too many correspondences exist, it selects plot_n evenly spaced ones
        if plot_n is not None:
            if xys.shape[0] > plot_n:
                ixs = torch.linspace(0, xys.shape[0]-1, plot_n).to(torch.int64)
                xys = xys[ixs, :]

        if lines:
            if color is not None:
                #Converts chosen color to an RGB tuple; LineCollection requires an rgb tuple
                color = mcolors.to_rgb(color)

            # yx convention; Uses matplotlib.collections.LineCollection to efficiently draw all the line segments
            plot = mplcollections.LineCollection(
                xys.numpy(),
                color=color,
                linewidth=linewidth
            )
            #Adds them to the existing axis
            self._ax.add_collection(plot)
        else:
            plot = None

        #Flattens all point coordinates into a list; Plots small while dots with black edges at every point location
        if marks:
            self._ax.scatter(
                xys[:, :, 0].numpy().flatten(),
                xys[:, :, 1].numpy().flatten(),
                marker='o',
                c='white',
                edgecolor='black',
                s=marker_size,
            )
    
        #Returns the LineCollection object (or None if no lines drawn)
        return plot