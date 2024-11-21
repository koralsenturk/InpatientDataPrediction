
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor



#load dataset
df = pd.read_csv("C:/Users/koralsenturk/AppData/Local/anaconda3/SagliktaYapayZeka/3_kodlar_verisetleri/4_SagliktaMakineOgrenmesiUygulamalari/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv")
df["Length of Stay"] = df["Length of Stay"].replace("120 +", 120)
df["Length of Stay"] = pd.to_numeric(df["Length of Stay"])
los = df["Length of Stay"]
# df_isna = df.isna().sum()
# print(df_isna)
for column in df.columns:
    unique_values = len(df[column].unique())
    # print(f"Number of unique values in {column}: {unique_values}")

df = df[df["Patient Disposition"] != "Expired"] #Bu veriden (iptal olmuş) kurtulmak için yazılmış kod




#EDA
"""
hasta yatiş süresi ile ilgili olan veriler;
type of admission
payment type

"""
# sns.boxplot (x = "Payment Typology 1", y = "Length of Stay", data = df)
# plt.title("Payment Typology 1 vs Length of Stay")
# plt.xticks(rotation = 60)

# sns.countplot(x = "Age Group", data = df[df["Payment Typology 1"] == "Medicare"])
# plt.title("Medicare Patients for Age Group")


# sns.boxplot (x = "Type of Admission", y = "Length of Stay", data = df)
# plt.title("Type of Admission vs Length of Stay")
# plt.xticks(rotation = 60)

sns.boxplot (x = "Age Group", y = "Length of Stay", data = df)
plt.title("Age Group vs Length of Stay")
plt.xticks(rotation = 60)
# plt.show()



#feature encoding- selection: label encoding
df = df.drop(["Hospital Service Area", 
"Hospital County", 
"Operating Certificate Number", 
"Facility Name", 
"Zip Code - 3 digits",
"Patient Disposition",
"Discharge Year",
"CCSR Diagnosis Description",
"CCSR Procedure Description",
"APR MDC Description",
"APR Severity of Illness Code",
"Payment Typology 2",
"Payment Typology 3",
"Birth Weight",
"Total Charges",
"Total Costs"],
axis = 1)

age_group_index = {"0 to 17":1, "18 to 29":2, "30 to 49":3, "50 to 69":4, "70 or Older":5}
gender_index = {"U":0, "F":1, "M":2}
risk_and_severity_index = {np.nan:0, "Minor":1, "Moderate":2, "Major":3, "Extreme":4}

df["Age Group"] = df["Age Group"].apply(lambda x: age_group_index[x])
df["Gender"] = df["Gender"].apply(lambda x: gender_index[x])
df["APR Risk of Mortality"] = df["APR Risk of Mortality"].apply(lambda x: risk_and_severity_index[x])

encoder = OrdinalEncoder()
df["Race"] = encoder.fit_transform(np.asarray(df["Race"]).reshape(-1, 1))
df["Ethnicity"] = encoder.fit_transform(np.asarray(df["Ethnicity"]).reshape(-1, 1))
df["Type of Admission"] = encoder.fit_transform(np.asarray(df["Type of Admission"]).reshape(-1, 1))
df["CCSR Diagnosis Code"] = encoder.fit_transform(np.asarray(df["CCSR Diagnosis Code"]).reshape(-1, 1))
df["CCSR Procedure Code"] = encoder.fit_transform(np.asarray(df["CCSR Procedure Code"]).reshape(-1, 1))
df["APR Medical Surgical Description"] = encoder.fit_transform(np.asarray(df["APR Medical Surgical Description"]).reshape(-1, 1))
df["Payment Typology 1"] = encoder.fit_transform(np.asarray(df["Payment Typology 1"]).reshape(-1, 1))
df["Emergency Department Indicator"] = encoder.fit_transform(np.asarray(df["Emergency Department Indicator"]).reshape(-1, 1))

# missing value kontrolu
df.isna().sum()

df = df.drop("CCSR Procedure Code", axis = 1)
df = df.dropna(subset=["Permanent Facility Id", "CCSR Diagnosis Code"])

# train test split
X = df.drop(["Length of Stay"], axis=1)
y = df["Length of Stay"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Regression: train ve test

dtree = DecisionTreeRegressor(max_depth=10)
dtree.fit(X_train, y_train)
train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print("RMSE: Train: ", np.sqrt(mean_squared_error(y_train, train_prediction)))
print("RMSE: Test: ", np.sqrt(mean_squared_error(y_test, test_prediction)))

"""
overfitting
RMSE: Train:  2.84783327422551 -> 7 - 10 - 13
RMSE: Test:  7.976502723219912 -> 2 - 10 - 18

after max_depth = 10
RMSE: Train:  6.088278470926022
RMSE: Test:  6.242028697402208
"""

# kategorik hale getirme: solve classification problem: train ve test 

bins = [0, 5, 10, 20, 30, 50, 120]
labels = [5, 10, 20, 30, 50, 120]

df["los_bin"] = pd.cut(x=df["Length of Stay"], bins=bins)
df["los_label"] = pd.cut(x=df["Length of Stay"], bins=bins, labels=labels)
df_ = df.head(50)
df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace(","," -"))
df["los_bin"] = df["los_bin"].apply(lambda x: str(x).replace("120","120+"))

f, ax = plt.subplots()
sns.countplot(x="los_bin", data=df)

new_X = df.drop(["Length of Stay", "los_bin", "los_label"], axis = 1)
new_y = df["los_label"]

X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size = 0.2, random_state = 42)

dtree = DecisionTreeClassifier(max_depth=10)
dtree.fit(X_train, y_train)

train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

print("Train Accuracy: ", accuracy_score(y_train, train_prediction))
print("Test Accuracy: ", accuracy_score(y_test, test_prediction))
print("Classification report: ", classification_report(y_test, test_prediction))

"""
Overfitting
Train Accuracy:  0.9244704097809807
Test Accuracy:  0.6851298279739233

after max_depth = 10
Train Accuracy:  0.7418194070663043
Test Accuracy:  0.7410697755934764
"""
