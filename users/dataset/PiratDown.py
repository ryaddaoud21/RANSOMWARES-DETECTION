import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle

#%%
def loadData():
    """
    This function load the dataset from csv file
    Returns two DataFrame the first is X and the second is the label Y
    -------
    df : DataFrame
        X loaded from the RecapV3.csv.
    outputDF : DataFrame
        Y the label.
    """
    """load data from CSV files"""
    df = pd.read_csv("recap.csv", index_col=False, sep=';')
    outputDF = pd.read_csv("out.csv", index_col=False, sep=';')
    ##outputDF.drop(axis=1, columns=outputDF.columns[1], inplace=True)
    print('Data loaded')
    print(df.shape)
    return df,outputDF
#%%
def normalizeDataSet(df):
    """use MinMaxScaler to normalize df."""
    scaler = MinMaxScaler()
    scaler.fit(df)
    normaLizedX = scaler.transform(df.values)
    dff = pd.DataFrame(normaLizedX)
    dff.columns = df.columns
    df = dff
    return df
#%%
def trainModels(df):
    """train model and show their accuracy."""
    #Train a Decision Tree
    rng = np.random.RandomState(1)
    DTC = DecisionTreeClassifier(max_depth=80)
    DTC.fit(X_train, y_train)
    
    #Train an AdaBoost with Decision Tree
    rng = np.random.RandomState(1)
    AdaBoostDT = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=24), n_estimators=100, random_state=rng)
    AdaBoostDT.fit(X_train, y_train)

    #Train a MLP model
    mlp = MLPClassifier(alpha=1e-5,
                        hidden_layer_sizes=(200, 100),
                        random_state=1,
                        activation="tanh",
                        max_iter=200)
    mlp.fit(X_train, y_train)
    
    
    #Train an SVM model
    svmModel = svm.LinearSVC()
    svmModel.fit(X_train, y_train)

    #Train a Random Forest
    rndmForest = RandomForestClassifier()
    rndmForest.fit(X_train, y_train)
    
    
    #Train a LogisticRegression
    clfLR = LogisticRegression(random_state=0).fit(X_train, y_train)   
    
    # saving models to disk
    pickle.dump(rndmForest, open('rf.sav', 'wb'))
    pickle.dump(clfLR, open('clfr.sav', 'wb'))
    pickle.dump(svmModel, open('svm.sav', 'wb'))
    pickle.dump(mlp, open('mlp.sav', 'wb'))
    pickle.dump(AdaBoostDT, open('AdaBoostDT.sav', 'wb'))
    pickle.dump(DTC, open('dtc.sav', 'wb'))


    
    
    print('Models Accuracy ---------------')
    print("DecisionTree", "{:.2f}".format(DTC.score(X_test, y_test)*100))
    print("AdaBoost DecitionTree", "{:.2f}".format(AdaBoostDT.score(X_test, y_test)*100))
    print("MLP", "{:.2f}".format(mlp.score(X_test, y_test)*100))
    print("RandomForestClassifier ",  "{:.2f}".format(rndmForest.score(X_test, y_test)*100))

    print("SVM: ",  "{:.2f}".format(svmModel.score(X_test, y_test)*100))
    print("Logistic Regression model : ",  "{:.2f}".format(clfLR.score(X_test, y_test)*100))
    print('------------------------------------')
#%%
df,outDF=loadData()
#%%
print("Normalizing dataset")
df=normalizeDataSet(df)
#split the dataset to 30% test and 70% training.
X_train, X_test, y_train, y_test = train_test_split(
    df.values, outDF.y, test_size=0.30, random_state=42)
trainModels(df)