import os
import pandas as pd
import numpy as np

os.getcwd()
os.chdir("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\digit-recognizer")

digit_train = pd.read_csv("train.csv")
digit_train.shape
digit_train.info()
digit_train.dtypes
digit_train.describe()

imageid= range(1,28001,1)
len(imageid)
label = np.random.randint(0,10,28000)
len(label)
dict = {'ImageId':imageid, 'Label':label}
out_df = pd.DataFrame.from_dict(dict)
out_df.shape
out_df.head(5)
out_df.set_index('ImageId', inplace=True)
out_df.to_csv("submission.csv")


