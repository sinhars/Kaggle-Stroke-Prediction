
#Plotter class for various data exploration techniques
from matplotlib.pyplot import title


class Visualizations():
    #Class Constructor
    def __init__(self, style="seaborn", fontSize=12):
        self.math = __import__("math")
        self.np = __import__("numpy")
        self.mpl = __import__("matplotlib")
        self.sns = __import__("seaborn")
        self.plt = self.mpl.pyplot

        params = {
            'axes.titlesize': fontSize+2,
            'xtick.labelsize': fontSize,
            'ytick.labelsize': fontSize
            }
        self.mpl.rcParams.update(params)
        self.plt.style.use(style)
        self.cmap = self.sns.diverging_palette(10, 250, s=90, l=33, as_cmap=True)

    #Method to plot counts/ histogram
    def plotCounts(self, xData, figSize, plotTitle):
        self.plt.figure(figsize=figSize)
        self.sns.countplot(x=xData)
        self.plt.title(plotTitle)
        self.plt.show()

    def plotHistograms(self, data, figSize, bins=50):
        data.hist(figsize=figSize, bins=bins)
        self.plt.show()

    def plotBoxPlot(self, data, x, y, hue, figSize, plotTitle):
        self.plt.figure(figsize=figSize)
        self.sns.boxplot(data=data, x=x, y=y, hue=hue)
        self.plt.title(plotTitle)
        self.plt.show()

    def plotMultipleBoxPlots(self, data, figSize, plotTitle):
        self.plt.figure(figsize=figSize)
        ax = self.sns.boxplot(data=data, orient="h")
        ax.set_title(plotTitle)
        ax.set(xscale="log")
        self.plt.show()

    def plotMultipleCountPlots(self, data, columns, figSize, plotTitle):
        ncols = self.math.floor(self.math.sqrt(len(columns)))
        nrows = self.math.ceil(len(columns) / ncols)
        fig = self.plt.figure(figsize=figSize, tight_layout=True)
        fig.suptitle(plotTitle)
        for i, col in enumerate(columns):
            fig.add_subplot(nrows, ncols, i + 1)
            self.sns.countplot(x=col, data=data)
            self.plt.xlabel("")
            self.plt.title(col)
        self.plt.show()

    def plotCorrelationMatrix(self, data, figSize, plotTitle):
        corrMatrix = data.corr()
        maskMatrix = self.np.triu(self.np.ones_like(corrMatrix, dtype=bool))
        self.plt.figure(figsize=figSize)
        self.sns.heatmap(corrMatrix, mask=maskMatrix, cmap=self.cmap, annot=False)
        self.plt.title(plotTitle)
        self.plt.show()
    
    def plotConfusionMatrix(self, data, labels, figSize, plotTitle):
        self.plt.figure(figsize=figSize)
        self.sns.heatmap(data, cmap=self.cmap, xticklabels=labels, yticklabels=labels, annot=True, fmt=".1%")
        self.plt.title(plotTitle)
        self.plt.show()
