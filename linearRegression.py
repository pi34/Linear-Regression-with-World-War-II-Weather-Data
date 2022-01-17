import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

weather_data = pd.read_csv("weather.csv")
weather_data.head()

sns.pairplot(weather_data)

plt.title("Housing Prices Pairplot")

X = weather_data[['MinTemp']]
y = weather_data[['MaxTemp']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

lreg = LinearRegression()

lr = lreg.fit(X_train, y_train)

print("Accuracy Score:", lr.score(X_test, y_test)*100)

sns.regplot(X, y, ci=None)
