import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def load_processed_data(path):
    """Carga los datos procesados."""
    return pd.read_csv(path)

def split_data(df):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X = df.drop(columns=['Price'])
    y = df['Price']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def build_pipelines():
    """Construye las canalizaciones de modelos."""
    categorical_cols = ['Airline', 'Source', 'Destination', 'Total_Stops']
    
    # Preprocesamiento para las columnas categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)],
        remainder='passthrough')

    # Canalizaciones para los modelos
    linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', LinearRegression())])

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', RandomForestRegressor(random_state=42))])

    xgboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])

    return linear_pipeline, rf_pipeline, xgboost_pipeline

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    """Entrena y evalúa un modelo dado."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

if __name__ == "__main__":
    # Ruta de los datos procesados
    path = 'data/processed/flight_data_processed.csv'

    # Cargar y dividir los datos
    df = load_processed_data(path)
    X_train, X_test, y_train, y_test = split_data(df)

    # Construir y evaluar los modelos
    linear_pipeline, rf_pipeline, xgboost_pipeline = build_pipelines()

    # Evaluación de la regresión lineal
    mse_linear, r2_linear = train_and_evaluate(linear_pipeline, X_train, y_train, X_test, y_test)
    print(f"Linear Regression MSE: {mse_linear}")
    print(f"Linear Regression R2: {r2_linear}")

    # Evaluación del Random Forest
    mse_rf, r2_rf = train_and_evaluate(rf_pipeline, X_train, y_train, X_test, y_test)
    print(f"Random Forest Regression MSE: {mse_rf}")
    print(f"Random Forest Regression R2: {r2_rf}")

    # Evaluación de XGBoost
    mse_xgb, r2_xgb = train_and_evaluate(xgboost_pipeline, X_train, y_train, X_test, y_test)
    print(f"XGBoost Regression MSE: {mse_xgb}")
    print(f"XGBoost Regression R2: {r2_xgb}")