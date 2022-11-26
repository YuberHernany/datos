import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('https://raw.githubusercontent.com/Kaushik-Varma/linear_regression_model_python/main/Company_data.csv')
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


advertisements_train = train_set[['TV', 'Radio', 'Newspaper']]
advertisements_test = test_set[['TV', 'Radio', 'Newspaper']]
sales_train = train_set['Sales']
sales_test = test_set['Sales']
lin_reg = LinearRegression()
lin_reg.fit(advertisements_train, sales_train)

# preparándose para el gráfico etapa de entrenamiento
x_min, x_max = advertisements_train['TV'].min(), advertisements_train['TV'].max()
y_min, y_max = advertisements_train['Radio'].min(), advertisements_train['Radio'].max()
newsp_min, newsp_max = advertisements_train['Newspaper'].min(), advertisements_train['Newspaper'].max()
newsp_mean = (newsp_min + newsp_max) / 2
X, Y = np.meshgrid(np.array([x_min, x_max]), np.array([y_min, y_max]))
instancias = np.array([[x_min, y_min, newsp_min],
                        [x_min, y_max, newsp_mean],
                        [x_max, y_min, newsp_mean], 
                        [x_max, y_max, newsp_max]])
Z = lin_reg.predict(instancias).reshape(2,2).T

# fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
# ax.scatter(advertisements_train['TV'], advertisements_train['Radio'], sales_train)
# ax.plot_surface(X, Y, Z, alpha=0.6)
# plt.show()

# preparándose para el gráfico etapa de testeo
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.scatter(advertisements_test['TV'], advertisements_test['Radio'], sales_test)
ax.plot_surface(X, Y, Z, alpha=0.6)
plt.show()