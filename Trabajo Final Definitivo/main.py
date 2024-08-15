import os
import pandas as pd

# Crear el directorio 'data/processed' si no existe
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

# Cargar los datos
flight_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1IYUWY3soT08zkge-RHNYHqlgv1B48tkFPAJDC0pi0cs/export?format=csv')

# Guardar los datos procesados
output_path = os.path.join(output_dir, 'flight_data_processed.csv')
flight_data.to_csv(output_path, index=False)

print(f"Archivo guardado en: {output_path}")

from etl_functions import load_data, explore_data, visualize_data, clean_data, save_processed_data
from ml_functions import load_processed_data, split_data, build_pipelines, train_and_evaluate

def main():
    # URL y ruta de archivos
    url = 'https://docs.google.com/spreadsheets/d/1IYUWY3soT08zkge-RHNYHqlgv1B48tkFPAJDC0pi0cs/export?format=csv'
    processed_data_path = 'data/processed/flight_data_processed.csv'

    # Proceso ETL
    flight_data = load_data(url)
    explore_data(flight_data)
    visualize_data(flight_data)
    flight_data = clean_data(flight_data)
    save_processed_data(flight_data, processed_data_path)

    # Modelado y evaluación
    df = load_processed_data(processed_data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    linear_pipeline, rf_pipeline, xgboost_pipeline = build_pipelines()

    # Evaluar modelos
    mse_linear, r2_linear = train_and_evaluate(linear_pipeline, X_train, y_train, X_test, y_test)
    print(f"Linear Regression MSE: {mse_linear}")
    print(f"Linear Regression R2: {r2_linear}")

    mse_rf, r2_rf = train_and_evaluate(rf_pipeline, X_train, y_train, X_test, y_test)
    print(f"Random Forest Regression MSE: {mse_rf}")
    print(f"Random Forest Regression R2: {r2_rf}")

    mse_xgb, r2_xgb = train_and_evaluate(xgboost_pipeline, X_train, y_train, X_test, y_test)
    print(f"XGBoost Regression MSE: {mse_xgb}")
    print(f"XGBoost Regression R2: {r2_xgb}")

if __name__ == "__main__":
    main()