
class Modelling():
    #Class Constructor
    def __init__(self, vis):
        self.np = __import__("numpy")
        self.pd = __import__("pandas")
        self.sklearn = __import__("sklearn")
        self.metrics = self.sklearn.metrics
        self.mod_select = self.sklearn.model_selection
        self.vis = vis

    def splitData(self, X, y, testSize, randomSeed, resetIndex=False):
        splits = self.mod_select.train_test_split(X, y, test_size=testSize, stratify=y, random_state=randomSeed)
        if resetIndex == True:
            splits = [df.reset_index(drop=True) for df in splits]
        return (splits)

    def preprocData(self, X, y, highCorrColumns, scalerModel, lowVarColumns):
        X_no_corrs = X.drop(columns=highCorrColumns)
        X_scaled = self.pd.DataFrame(scalerModel.transform(X_no_corrs))
        X_scaled.columns = X_no_corrs.columns
        X_preproc = X_scaled.drop(columns=lowVarColumns)
        return (X_preproc, y)

    #Method to plot counts/ histogram
    def getModelPerformance(self, trueVals, preds, figSize, plotTitle, targetNames):
        confMatrix = self.metrics.confusion_matrix(trueVals, preds, normalize="true")
        report = self.metrics.classification_report(trueVals, preds, target_names=targetNames)
        print("F1-score = %.2f%% \n" % (self.metrics.f1_score(trueVals, preds) * 100))
        print(report)
        self.vis.plotConfusionMatrix(confMatrix, labels=targetNames, figSize=figSize, plotTitle=plotTitle)