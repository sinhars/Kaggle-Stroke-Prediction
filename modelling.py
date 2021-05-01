import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, classification_report, f1_score

class Modelling():
    #Class Constructor
    def __init__(self, vis):
        self.vis = vis

    def splitData(self, X, y, testSize, randomSeed, resetIndex=False):
        splits = train_test_split(X, y, test_size=testSize, stratify=y, random_state=randomSeed)
        if resetIndex == True:
            splits = [df.reset_index(drop=True) for df in splits]
        return (splits)

    def oneHotEncodeData(self, X, categoryColumns, encoderModels=None):
        X_enc = X.copy(deep=True)
        if encoderModels is None:
            encoderModels = {}
        for col in categoryColumns:
            if col not in encoderModels.keys():
                colEncoder = OneHotEncoder(handle_unknown="ignore")
                colEncoder.fit(X_enc[[col]])
                encoderModels[col] = colEncoder
            else:
                colEncoder = encoderModels[col]
            encData = pd.DataFrame(colEncoder.transform(X[[col]]).toarray(), columns=np.array(colEncoder.categories_).flatten())
            X_enc = pd.concat([X_enc, encData], axis=1)
            X_enc.drop(columns=col, inplace=True)
        return (X_enc, encoderModels)

    def imputeData(self, X, imputerModel=None):
        if imputerModel is None:
            imputerModel = KNNImputer()
            imputerModel.fit(X)
        imputedData = imputerModel.transform(X)
        X_imp = pd.DataFrame(imputedData, columns=X.columns)
        return(X_imp, imputerModel)
    
    def scaleData(self, X, numericColumns, scalerModel=None):
        if scalerModel is None:
            scalerModel = RobustScaler()
            scalerModel.fit(X[numericColumns])
        scaledData = pd.DataFrame(scalerModel.transform(X[numericColumns]), columns=numericColumns)
        X_scaled = X.drop(columns=numericColumns)
        X_scaled = pd.concat([X_scaled, scaledData], axis=1)
        return (X_scaled, scalerModel)

    def preprocData(self, X, categoryColumns, numericColumns, encoderModels=None, imputerModel=None, scalerModel=None):
        X_enc, encoderModels = self.oneHotEncodeData(X=X, categoryColumns=categoryColumns, encoderModels=encoderModels)
        X_imp, imputerModel = self.imputeData(X=X_enc, imputerModel=imputerModel)
        X_scaled, scalerModel = self.scaleData(X=X_imp, numericColumns=numericColumns, scalerModel=scalerModel)
        return (X_scaled, encoderModels, imputerModel, scalerModel)

    #Method to plot counts/ histogram
    def getModelPerformance(self, trueVals, preds, figSize, plotTitle, targetNames):
        confMatrix = confusion_matrix(trueVals, preds, normalize="true")
        report = classification_report(trueVals, preds, target_names=targetNames)
        print("F1-score = %.2f%% \n" % (f1_score(trueVals, preds) * 100))
        print(report)
        self.vis.plotConfusionMatrix(confMatrix, labels=targetNames, figSize=figSize, plotTitle=plotTitle)