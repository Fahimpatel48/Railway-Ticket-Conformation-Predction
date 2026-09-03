import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/Railway Ticket Confirmation.csv")
le_from = LabelEncoder()
le_To = LabelEncoder()
le_Class = LabelEncoder()
le_Quota = LabelEncoder()
le_Status = LabelEncoder()
le_Special = LabelEncoder()
le_target = LabelEncoder()
df["Special Considerations"] = df["Special Considerations"].fillna("None")

#LabelEncoder
df["From"] = le_from.fit_transform(df["From"])
df["To"] = le_To.fit_transform(df["To"])
df["Class of Travel"] = le_Class.fit_transform(df["Class of Travel"])
df["Quota"] = le_Quota.fit_transform(df["Quota"])
df["Current Status"] = le_Status.fit_transform(df["Current Status"])
df["Special Considerations"] = le_Special.fit_transform(df["Special Considerations"])
df["Confirmation Status"] = le_target.fit_transform(df["Confirmation Status"])

'''df["Date of Journey"] = pd.to_datetime(df["Date of Journey"],dayfirst=True)
print(df["Date of Journey"].head())

df["Booking Date"] = pd.to_datetime(df["Booking Date"],dayfirst=True)
print(df["Booking Date"].head())'''


x = df[["From","To","Class of Travel","Quota","Current Status","Number of Passengers","Special Considerations"]]
y = df["Confirmation Status"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

'''print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)'''

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred_class = np.round(y_pred).astype(int)

r2 = model.score(x_test, y_test)
print("R2: ",r2)

prediction = model.predict(x_test)
prediction = np.clip(prediction, 0, 1)
percentage = prediction*100

print(le_target.classes_)

# User Input

print("\nFROM: ",le_from.classes_)
from_input = input("From: ")
print("\nTO: ",le_To.classes_)
to_input = input("TO: ")
print("\n",le_Class.classes_)
class_input = input("Class of Travel: ")
print("\n",le_Quota.classes_)
quota_input = input("Quota: ")
print("\n",le_Status.classes_)
status_input = input("Current Status: ")
passengers_input = int(input("Number of Passengers: "))
print("\n",le_Special.classes_)
special_input = input("Special Consideration: ")

if special_input.lower() == "none":
    special_input = "None"
# Conver into Number
from_value = le_from.transform([from_input])[0]
to_value = le_To.transform([to_input])[0]
class_value = le_Class.transform([class_input])[0]
qouta_value = le_Quota.transform([quota_input])[0]
status_value = le_Status.transform([status_input])[0]


special_value = le_Special.transform([special_input])[0]

# New Ticket Data

new_ticket = np.array([[
    from_value,
    to_value,
    class_value,
    qouta_value,
    status_value,
    passengers_input,
    special_value
]])

# prediction
prediction = model.predict(new_ticket)
prediction = np.clip(prediction, 0, 1)
percentage = prediction[0] * 100
print("\n confirmation Prediction is : ",round(percentage, 2), "%")