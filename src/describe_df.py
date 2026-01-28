import pandas as pd

def describe_df(df):
    """
    Genera un resumen del DataFrame con información relevante para cada columna.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.

    Retorna:
    pd.DataFrame: DataFrame resumen con:
        - Tipo de variable
        - % de valores nulos
        - Nº de valores únicos
        - % de cardinalidad (valores únicos / total de filas)
    """
    resumen = pd.DataFrame({
        col: [
            df[col].dtype,
            100 * df[col].isnull().mean(),
            df[col].nunique(),
            100 * df[col].nunique() / len(df)
        ]
        for col in df.columns
    }, index=["Tipo", "% Nulos", "Valores únicos", "% Cardinalidad"])
    return resumen