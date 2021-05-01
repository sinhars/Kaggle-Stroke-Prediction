import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 

class Visualizations():
    #Class Constructor
    def __init__(self, style="seaborn", fontSize=12):
        params = {
            'axes.titlesize': fontSize+2,
            'xtick.labelsize': fontSize,
            'ytick.labelsize': fontSize
            }
        mpl.rcParams.update(params)
        plt.style.use(style)
        self.cmap = sns.diverging_palette(10, 250, s=90, l=33, as_cmap=True)

    #Method to plot counts/ histogram
    def plotCounts(self, xData, figSize, plotTitle):
        plt.figure(figsize=figSize)
        sns.countplot(x=xData)
        plt.title(plotTitle)
        plt.show()

    def plotHistograms(self, data, figSize, bins=50):
        data.hist(figsize=figSize, bins=bins)
        plt.show()

    def plotBoxPlot(self, data, x, y, hue, figSize, plotTitle):
        plt.figure(figsize=figSize)
        sns.boxplot(data=data, x=x, y=y, hue=hue)
        plt.title(plotTitle)
        plt.show()

    def plotMultipleBoxPlots(self, data, figSize, plotTitle):
        plt.figure(figsize=figSize)
        ax = sns.boxplot(data=data, orient="h")
        ax.set_title(plotTitle)
        ax.set(xscale="log")
        plt.show()

    def plotMultipleCountPlots(self, data, columns, figSize, plotTitle):
        ncols = math.floor(math.sqrt(len(columns)))
        nrows = math.ceil(len(columns) / ncols)
        fig = plt.figure(figsize=figSize, tight_layout=True)
        fig.suptitle(plotTitle)
        for i, col in enumerate(columns):
            fig.add_subplot(nrows, ncols, i + 1)
            sns.countplot(x=col, data=data)
            plt.xlabel("")
            plt.title(col)
        plt.show()

    def plotCorrelationMatrix(self, data, figSize, plotTitle):
        corrMatrix = data.corr()
        maskMatrix = np.triu(np.ones_like(corrMatrix, dtype=bool))
        plt.figure(figsize=figSize)
        sns.heatmap(corrMatrix, mask=maskMatrix, cmap=self.cmap, annot=False)
        plt.title(plotTitle)
        plt.show()
    
    def plotConfusionMatrix(self, data, labels, figSize, plotTitle):
        plt.figure(figsize=figSize)
        sns.heatmap(data, cmap=self.cmap, xticklabels=labels, yticklabels=labels, annot=True, fmt=".1%")
        plt.title(plotTitle)
        plt.show()
