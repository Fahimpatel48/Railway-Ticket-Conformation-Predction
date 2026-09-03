import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("dataset/Book1.csv")

print(df.head())
print(df.info())
print(df.describe())
plt.scatter(df["hour"],df["marks"])
plt.xlabel("hour")
plt.ylabel("marks")
plt.title("hour vs marks")
plt.show()

print("coviriance")
print(df.cov())

print(df.corr())

x=df[["hour"]]
y=df["marks"]
model=LinearRegression()
model.fit(x,y)
m=model.coef_[0]
b=model.intercept_
print(f"Equation: y={m:.2f}x+{b:.2f}")

pred = model.predict([[4]])
print("Marks for 4 hour: ",pred[0])

r2 = model.score(x,y)
print("r2 value: ",r2)

y_pred = model.predict(x)
rss = np.sum((y-y_pred)**2)
print("rss: ",rss)

plt.scatter(x,y)
plt.plot(x,y_pred)
plt.xlabel("hour")
plt.ylabel("marks")
plt.title("Regression Line")
plt.show()
