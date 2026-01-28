def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Sugiere el tipo de variable para cada columna de un DataFrame según su cardinalidad.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    umbral_categoria (int): Umbral máximo de valores únicos para considerar una variable como categórica.
    umbral_continua (float): Umbral mínimo de % de cardinalidad para considerar una variable como numérica continua.

    Retorna:
    pd.DataFrame: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'.
    """
    resultado = []
    n = len(df)
    for col in df.columns:
        cardinalidad = df[col].nunique()
        porcentaje_card = cardinalidad / n
        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif porcentaje_card >= umbral_continua:
            tipo = "Numerica Continua"
        else:
            tipo = "Numerica Discreta"
        resultado.append({"nombre_variable": col, "tipo_sugerido": tipo})
    return pd.DataFrame(resultado)