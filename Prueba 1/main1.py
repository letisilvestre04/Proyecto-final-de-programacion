# Importar todas las librerias pertinentes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# Cargar datos
flight_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1IYUWY3soT08zkge-RHNYHqlgv1B48tkFPAJDC0pi0cs/export?format=csv')
flight_data.head()

# Exploración y Análisis Preliminar

flight_data.describe()
flight_data.info()
flight_data["Source"].value_counts()

# Visualización del comportamiento de las variables con la varible de interés (Price)

# Distribución de una columna numérica, Price
sns.histplot(flight_data['Price'].dropna(), kde=True)
plt.title('Distribución del precio')
plt.show()

# Distribución de una variable categórica, por ejemplo, 'Flight Status'
sns.countplot(x='Total_Stops', data=flight_data)
plt.title('Distribución de paradas ')
plt.show()

# Gráfico Price vs. Month
plt.figure(figsize=(10, 6))
sns.lineplot(x= 'Month', y= 'Price', data= flight_data, marker='o')
plt.title('Relación entre Precio y Mes')
plt.xlabel('Month')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Gráfico Price vs. Dep_hours
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Dep_hours', y='Price', data= flight_data, hue='Airline', s=100)
plt.title('Relación entre Price y Dep_hours')
plt.xlabel('Departure Hours')
plt.ylabel('Price')
plt.xticks(range(0, 24, 1))  # Horas de 0 a 23 con intervalos de 1
plt.grid(True)
plt.show()

# Limpieza y transfromación de datos
flight_data.dropna(inplace=True)
flight_data.isnull().sum()

#Guardar los datos transformados y procesados
flight_data.to_csv('flight_data_processed.csv', index=False)

# Modelado de Regresión

# Selección del target
X = flight_data.drop(columns=['Price'])
y = flight_data['Price']

# Columnas categóricas para codificar
categorical_cols = ['Airline', 'Source', 'Destination', 'Total_Stops']

# Canalización de preprocesamiento para características categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)],
    remainder='passthrough')

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Modelo de regresión lineal
# Crear una canalización que incluya preprocesamiento y el modelo de regresión lineal
linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', LinearRegression())])

# Prueba 1

# Entrenando el modelo
linear_pipeline.fit(X_train, y_train)

# Predecir los precios en el equipo de prueba.
y_pred_linear = linear_pipeline.predict(X_test)

# Evaluación
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression MSE: {mse_linear}")
print(f"Linear Regression R2: {r2_linear}")

# Prueba 2
# 2. Modelo de regresión de bosque aleatorio
# Crear una canalización que incluya preprocesamiento y el modelo de regresión Random Forest
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(random_state=42))])

# Entrenamiento
rf_pipeline.fit(X_train, y_train)

# Predecir los precios en el equipo de prueba.
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluación
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regression MSE: {mse_rf}")
print(f"Random Forest Regression R2: {r2_rf}")

#Prueba 3
# Crear una canalización que incluya preprocesamiento y el modelo XGBoost
xgboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])

# Entrenar el modelo
xgboost_pipeline.fit(X_train, y_train)

# Predecir los precios en el conjunto de prueba
y_pred_xgb = xgboost_pipeline.predict(X_test)

# Evaluar el modelo
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Regression MSE: {mse_xgb}")
print(f"XGBoost Regression R2: {r2_xgb}")

# Prueba con el usuario para confirmar que el modelo fue entrenado correctamente
def predict_flight_price(rf_pipeline): 
    # Entradas del usuario
    airline = input("Enter the Airline: ")
    source = input("Enter the Source City: ")
    destination = input("Enter the Destination City: ")
    total_stops = int(input("Enter the Total Stops (e.g., 0 for non-stop, 1 for one stop, etc.): "))
    date = int(input("Enter the Date of Journey (day of the month): "))
    month = int(input("Enter the Month of Journey (numerical): "))
    year = int(input("Enter the Year of Journey: "))
    dep_hours = int(input("Enter the Departure Hour (24-hour format): "))
    dep_min = int(input("Enter the Departure Minute: "))
    arrival_hours = int(input("Enter the Arrival Hour (24-hour format): "))
    arrival_min = int(input("Enter the Arrival Minute: "))
    duration_hours = int(input("Enter the Duration Hours: "))
    duration_min = int(input("Enter the Duration Minutes: "))


    # Crear un DataFrame para la entrada del usuario 
    input_data = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [destination],
        'Total_Stops': [total_stops],
        'Date': [date],
        'Month': [month],
        'Year': [year],
        'Dep_hours': [dep_hours],
        'Dep_min': [dep_min],
        'Arrival_hours': [arrival_hours],
        'Arrival_min': [arrival_min],
        'Duration_hours': [duration_hours],
        'Duration_min': [duration_min]
    })

    # Predecir el precio
    predicted_price = rf_pipeline.predict(input_data)
    print(f"The predicted flight price is: {predicted_price[0]:.2f}")

# Ejecute la función para predecir el precio del vuelo.
predict_flight_price(rf_pipeline)

