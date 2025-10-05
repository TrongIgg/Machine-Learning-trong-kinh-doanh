import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# from sklearn.externals import joblib

cars_df = pd.read_csv("nhap/cars_dataset.csv", sep="\t", encoding='utf-8')

num_cols = ['price', 'length', 'height', 'width', 'weight', 'weightTotal', 'emissionsCO2', 'numberOfAxles',
            'numberOfDoors', 'numberOfForwardGears', 'seatingCapacity', 'cargoVolume', 'roofLoad',
            'accelerationTime', 'fuelCapacity', 'fuelConsumption', 'speed', 'payload', 'trailerWeight',
            'vEengineDisplacement', 'vEenginePower', 'torque']

cat_cols = []
for col in cars_df.columns:
    if not (col in num_cols):
        cat_cols.append(col)

print(len(num_cols), len(cat_cols))
print(num_cols, '\n', cat_cols)

df = cars_df.copy()


# fucntion convert cac thuoc tinh khac
def cvtFloat(x):
    if type(x) == str:
        temp = x.replace(',', '.').split()[0]
    else:
        temp = x
    val = None
    try:
        val = float(temp)
    except ValueError:
        return val
    return val


for el in num_cols:
    if el != 'cargoVolume':
        print(el)
        df[el] = df[el].apply(cvtFloat)


def cvtFloat_cargoVolume(x):
    temp = x.split()[0]
    temp = temp.replace('-', ' ')
    temp = temp.split()
    if len(temp) > 0:
        temp = temp[-1]
    else:
        temp = x
    val = None
    try:
        val = float(temp)
    except ValueError:
        return val
    return val


df['cargoVolume'] = df['cargoVolume'].apply(cvtFloat_cargoVolume)

df[num_cols].info()
for cat in cat_cols[:]:
    print(cat, len(cars_df[cat].unique()))

'''
* Có thể loại bỏ cột vehicleTransmission vì chỉ có 1 giá trị, không có ý nghĩa trong việc học.
* Cột fuelType và vEfuelType là giống nhau (do quá trình crawl nhóm không để ý), có thể drop cột fuelType.
* Các cột url, name, model có nhiều ý nghĩa, nên có thể loại bỏ.
* brand có thể xét vì có tới 89 giá trị (có khả năng sẽ có ý nghĩa với các brand có giá trị cao), modelDate cần xem xét.

**=> Số cột còn lại là: eLabel (9), bodyType (11), driveWheelConfiguration (6), vEengineType (4), vEfuelType (11).**
'''


# chuẩn hóa cột modelDate
def norm_modelDate(x):
    if (x == 0):
        return None
    else:
        return str(x)


df['modelDate'] = df['modelDate'].apply(norm_modelDate)
df['modelDate'].unique()
df['driveWheelConfiguration'].unique()
df['bodyType'].unique()
df['eLabel'].unique()
df['vEengineType'].unique()
df['vEfuelType'].unique()

'''
* Cột driveWheelConfiguration không có giá trị lỗi ('N.A.', '-', ...)
* Các cột bodyType, vEengineType, vEfuelType có chứa nan (đã được xử lý).
* Cột eLabel có chứa các giá trị lỗi, cần được chuẩn hóa. Sau khi chuẩn hóa, dòng thiếu dữ liệu quá nhiều nên cần loại bỏ khi qua bước xử lý.
'''


def norm_eLabel(x):
    if (x == 'N.A.' or x == '-'):
        return None
    else:
        return x


df['eLabel'] = df['eLabel'].apply(norm_eLabel)
df[cat_cols].info()

df.to_csv("cars_preprocessed_undrop.csv", sep="\t", index=False, encoding='utf-8')
