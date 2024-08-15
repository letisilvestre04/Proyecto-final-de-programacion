import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(url):
    """Carga los datos desde la URL proporcionada."""
    flight_data = pd.read_csv(url)
    return flight_data

def explore_data(df):
    """Explora y analiza los datos preliminarmente."""
    print(df.describe())
    print(df.info())
    print(df["Source"].value_counts())

def visualize_data(df):
    """Visualiza la distribución de datos y relaciones."""
    # Distribución de 'Price'
    sns.histplot(df['Price'].dropna(), kde=True)
    plt.title('Distribución del precio')
    plt.show()

    # Distribución de 'Total_Stops'
    sns.countplot(x='Total_Stops', data=df)
    plt.title('Distribución de paradas')
    plt.show()

    # Price vs. Month
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Price', data=df, marker='o')
    plt.title('Relación entre Precio y Mes')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

    # Price vs. Dep_hours
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Dep_hours', y='Price', data=df, hue='Airline', s=100)
    plt.title('Relación entre Price y Dep_hours')
    plt.xlabel('Departure Hours')
    plt.ylabel('Price')
    plt.xticks(range(0, 24, 1))
    plt.grid(True)
    plt.show()

def clean_data(df):
    """Limpia y transforma los datos."""
    df.dropna(inplace=True)
    return df

def save_processed_data(df, path):
    """Guarda los datos transformados en un archivo CSV."""
    df.to_csv(path, index=False)

if __name__ == "__main__":
    # URL del dataset
    url = 'https://docs.google.com/spreadsheets/d/1IYUWY3soT08zkge-RHNYHqlgv1B48tkFPAJDC0pi0cs/export?format=csv'
    
    # Ejecutar las funciones ETL
    flight_data = load_data(url)
    explore_data(flight_data)
    visualize_data(flight_data)
    clean_data(flight_data)
    save_processed_data(flight_data, 'data/processed/flight_data_processed.csv')