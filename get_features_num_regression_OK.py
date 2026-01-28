

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve las columnas numéricas de un DataFrame cuya correlación con una
    columna objetivo supera un umbral definido, y opcionalmente un test de significación.

    Argumentos:
    df (pd.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo (variable dependiente).
                      Debe ser numérica y de alta cardinalidad.
    umbral_corr (float): Valor entre 0 y 1 que indica el umbral mínimo de
                         correlación absoluta.
    pvalue (float, opcional): Nivel de significación estadística. Si es distinto
                              de None, se filtran solo las variables cuya
                              significación sea mayor o igual a 1 - pvalue.

    Devuelve:
    list: Lista de nombres de columnas numéricas que cumplen las condiciones.
          Devuelve None si alguna comprobación falla.
    """

    # ---------- COMPROBACIONES DE ENTRADA ----------

    # Comprobar que df es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un pandas DataFrame.")
        return None

    # Comprobar que target_col existe en el DataFrame
    if target_col not in df.columns:
        print("Error: target_col no existe en el DataFrame.")
        return None

    # Comprobar que target_col es numérica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser una variable numérica.")
        return None

    # Comprobar que umbral_corr es float entre 0 y 1
    if not isinstance(umbral_corr, float) or not (0 <= umbral_corr <= 1):
        print("Error: umbral_corr debe ser un float entre 0 y 1.")
        return None

    # Comprobar pvalue si se ha introducido
    if pvalue is not None:
        if not isinstance(pvalue, float) or not (0 < pvalue < 1):
            print("Error: pvalue debe ser None o un float entre 0 y 1.")
            return None

    # ---------- SELECCIÓN DE COLUMNAS NUMÉRICAS ----------

    columnas_numericas = df.select_dtypes(include="number").columns

    # Eliminar la columna target de la lista
    columnas_numericas = columnas_numericas.drop(target_col)

    # ---------- CÁLCULO DE CORRELACIONES ----------

    columnas_seleccionadas = []

    for col in columnas_numericas:
        # Eliminar filas con valores nulos
        data = df[[target_col, col]].dropna()

        # Necesitamos al menos 2 valores para correlación
        if len(data) < 2:
            continue

        corr, p_val = pearsonr(data[target_col], data[col])

        # Comprobar umbral de correlación
        if abs(corr) >= umbral_corr:
            # Si no se usa pvalue, se añade directamente
            if pvalue is None:
                columnas_seleccionadas.append(col)
            else:
                # Comprobar significación estadística
                if p_val <= pvalue:
                    columnas_seleccionadas.append(col)

    return columnas_seleccionadas