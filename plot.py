import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

path = "C:/Users/Junyi Chen/Desktop/NeuralKart-master - Copy/recordings/LR/TT/"
files=os.listdir(path)

ppp=[]
final_df=[]
for f in files:
	if f.find('search')>-1:
		l=os.listdir(path+f)
		l=[path+f+'/'+ x for x in l if x.find('txt')>-1]
		ppp.extend(l)

for steer in ppp:
	df=pd.read_csv(steer,sep=" ",header=None)
	final_df.append(df)

final_df=pd.concat(final_df)
print(final_df[0])
print(final_df[0].value_counts())
final_df.hist(bins=40)

plt.savefig('C:/Users/Junyi Chen/Desktop/NeuralKart-master - Copy/figure.png')

