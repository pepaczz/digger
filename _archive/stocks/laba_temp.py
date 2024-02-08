# import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# vytvoř umělý dataset
# proměnná y je závislá na druhé mocnině proměnné x a na šumu
x = np.linspace(-4, 6, 100)
noise = np.random.uniform(-5, 5, 100)
y = x**2 + noise



# vytvoř model lineární regrese pracující pouze s x v první mocnině
lm1 = LinearRegression()
lm1.fit(x.reshape(-1, 1), y)
print(lm1.coef_, lm1.intercept_)


# zobraz původní data a přímku vytvořenou modelem
plt.scatter(x, y)
plt.plot(x, lm1.predict(x.reshape(-1, 1)), color='red')
plt.show()


# vytvoř model lineární regrese pracující s x v první i druhé mocnině
# druhou mocninu vytvoř pomocí funkce PolynomialFeatures
lm2 = LinearRegression()
poly = PolynomialFeatures(degree=2)
lm2.fit(poly.fit_transform(x.reshape(-1, 1)), y)
print(lm2.coef_, lm2.intercept_)

# zobraz původní data a přímku vytvořenou modelem
plt.scatter(x, y)
plt.plot(x, lm2.predict(poly.fit_transform(x.reshape(-1, 1))), color='red')
plt.show()
