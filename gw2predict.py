import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as mp

while True:
    try:
        itemid = int(input("Please type the item ID. example for ectoplasm is: 19721 \n"))
    except ValueError:
        print("This is not a number, Please type the item ID")
        continue
    else:
        break

#pulling the csv from silver's API
url = "https://api.datawars2.ie/gw2/v2/history/csv?itemID=" + str(itemid)

while True:
    try:
        forecast = int(input("how many days do want to predict the prices in? Example: 20 \n"))
    except:
        print("This is not a number, Please type the number of days")
        continue
    else:
        break

while True:
    try:
        price = int(input("Type the actual price of the item you want to predict in coppers. Example: 19800 \n"))
    except:
        print("This is not a number, Please type the price in digits in copper")
        continue
    else:
        break

data = pd.read_csv(url)
data = data[["sell_price_max"]]
data["prediction"] = data[["sell_price_max"]].shift(-forecast)

x = np.array(data.drop(["prediction"], 1))
x = x[:-forecast]

y = np.array(data['prediction'])
y = y[:-forecast].reshape(-1,1)

# mp.scatter(x,y)
# mp.xlabel("Price")
# mp.ylabel("30 day Prediction")
# mp.show()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

linear = LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Accuracy: ",round(acc*100), "% using Linear Regression")

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)
acc = lasso_model.score(x_test, y_test)
print("Accuracy: ",round(acc*100), "% using Lasso Regression")

r = Ridge(alpha=0.1)
r.fit(x_train, y_train)
acc = r.score(x_test, y_test)
print("Accuracy: ",round(acc*100), "% using Ridge Regression")

e = ElasticNet(alpha=0.1)
e.fit(x_train, y_train)
acc = e.score(x_test, y_test)
print("Accuracy: ",round(acc*100), "% using Elastic Net Regression")

predict = np.array([price]).reshape(-1,1)
x = linear.predict(predict)
print("the predicted price of your item in ",forecast, " days is ",x[0][0])