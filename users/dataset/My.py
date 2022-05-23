import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



loaded_model = pickle.load(open('rf.sav', 'rb'))
df = pd.read_csv("recap.csv", index_col=False, sep=';')



scaler = MinMaxScaler()
scaler.fit(df)
normaLizedX = scaler.transform(df.values)
dff = pd.DataFrame(normaLizedX)
dff.columns = df.columns
df = dff


print(loaded_model.predict(df))
