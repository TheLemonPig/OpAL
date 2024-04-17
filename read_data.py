import numpy as np
import os
import pandas as pd

datapath = os.path.join(os.getcwd(),'Data')
for file in os.listdir(datapath):
    filepath = os.path.join(datapath,file)
    df = pd.read_csv(filepath)
    # array = np.genfromtxt(filepath, delimiter=',', encoding='utf-8')
    print(df)

