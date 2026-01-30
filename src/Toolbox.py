
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.stats import f_oneway
from scipy import stats

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
            100 * df[col].isnull().sum() / len(df),
            df[col].nunique(),
            100 * df[col].nunique() / len(df)
        ]
        for col in df.columns
    }, index=["Tipo", "% Nulos", "Valores únicos", "% Cardinalidad"])
    return resumen


def tipifica_variables(df, umbral_categoria, umbral_continua):
    """
    Genera un Dataframe con 2 columnas: 'nombre_variable' y 'tipo_sugerido',
    que tendrá tantas filas como variables tenga el DataFrame de entrada.

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
        porcentaje_card = 100 * ( cardinalidad / n )
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

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Analiza y visualiza la relación entre variables numéricas y un target 
    mediante filtros estadísticos de correlación y significancia.
    """
    
    # 1: VALIDACIÓN DE INTEGRIDAD DE DATOS
    # Comprobamos que el dataframe existe y el target es válido para regresión

    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame.")
        return None
    
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: '{target_col}' debe ser una columna numérica del dataframe.")
        return None
    
    # 2: SELECCIÓN ESTRATÉGICA DE COLUMNAS
    # Si la lista viene vacía, tomamos todas las numéricas por defecto

    if len(columns) == 0:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        if target_col in columns: columns.remove(target_col)
    else:
        # Si hay lista, filtramos solo las que realmente existen y son numéricas
        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and c != target_col]

    # 3: FILTRADO ESTADÍSTICO (CORRELACIÓN Y P-VALUE)
    selected_columns = []
    # Eliminamos nulos para el cálculo estadístico (crítico en datasets como Titanic)
    df_clean = df.dropna(subset=[target_col] + columns) 

    for col in columns:
        # Calculamos correlación de Pearson y su nivel de confianza (p-value)
        corr_val, p_val = pearsonr(df_clean[col], df_clean[target_col])
        
        # Filtro de fuerza: ¿La relación es lo suficientemente fuerte?
        if abs(corr_val) > umbral_corr:
            # Filtro de confianza: ¿La relación es estadísticamente significativa?
            if pvalue is None or p_val < pvalue:
                selected_columns.append(col)

    # Si tras los filtros no queda nada, avisamos al usuario
    if not selected_columns:
        print("No se encontraron variables que superen los umbrales estadísticos establecidos.")
        return []

    # 4: VISUALIZACIÓN DINÁMICA (MAX 5 VARS POR GRÁFICO)
    # Para evitar gráficos ilegibles, dividimos la lista en grupos de 4 + target
    max_vars = 4 
    for i in range(0, len(selected_columns), max_vars):
        chunk = selected_columns[i : i + max_vars]
        cols_to_plot = [target_col] + chunk
        
        # Generamos el pairplot (gráficos de dispersión y distribuciones)
        sns.pairplot(df[cols_to_plot], diag_kind='kde', corner=True)
        plt.suptitle(f"Análisis de Correlación: {chunk}", y=1.02)
        plt.show()

    return selected_columns


def get_features_cat_regression(dataframe, target_col, pvalue, usar_metrica = False):

    """
    Esta funcion devuelve columnas categoricas con relacion significativa respecto al target

    Variables: 

    dataframe
    target_col = Nombre de la columna target
    pvalue = Nivel de significancia (segun el ejercicio, su valor por defecto sera 0.5)
    usar_metrica: Para no hardcodearlo, me he permitido la libertad de añadir un parametro mas a la función

    Devuelve una lista de las columnas con relacion significativa
    """

    columnas_categoricas = []
    for columna in dataframe.columns:
       
        if columna == target_col: # Aqui identificamos columnas, omitiendo la target
            continue

        if dataframe[columna].dtype in ['object', 'category']:
            columnas_categoricas.append(columna) 
        elif dataframe[columna].dtype == "bool":
            columnas_categoricas.append(columna) #En estas lineas comprobamos si la columna es un dtype objeto o categoria, asi como un booleano
    if len(columnas_categoricas) == 0:
        print("No hay columnas categoricas en el dataframe")
        return []
    
    columnas_significativas = []

    for columna in columnas_categoricas:
        temp_df = dataframe[[columna, target_col]].dropna() #eliminamos nulos para evitar problemas

        categorias = temp_df[columna].unique()
        num_categorias = len(categorias)

        if num_categorias <2: #Checkeamos que hayan suficientes categorias
            continue
        
        grupos = [temp_df[temp_df[columna] == col][target_col].values
                  for col in categorias]
        grupos = [g for g in grupos if len(g) >= 2]

        if len(grupos) < 2:
            continue

        try:
            if  num_categorias == 2:
                if usar_metrica:
                    estadistica, p_val = stats.ttest_ind(grupos[0], grupos[1])
                else:
                    estadistica, p_val = stats.mannwhitneyu(grupos[0], grupos[1],
                                                            alternative= "two-sided")
            else:
                if usar_metrica:
                    estadistica, p_val = stats.f_oneway(*grupos) 
                else:
                    estadistica, p_val = stats.kruskal(*grupos)
            if p_val < pvalue:
                columnas_significativas.append(columna) #Pasamos por el t de student y mannwhitney pero en el caso de este dataframe nos encontraremos que hay mas de dos categorias, por tanto no son validos, hay que usar otros
        except Exception as e:
            print(f"Error procesado columna {columna}; {e}")
            continue
    return columnas_significativas


def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    
    """
    Genera un resumen del DataFrame con información relevante para cada columna.

    Argumentos:
    df (pd.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo con valor por defecto "".
    columns (list): Lista de nombres de columnas categóricas a analizar. 
    Si está vacía, se analizarán todas las columnas categóricas excepto la columna objetivo.
    pvalue (float): pvalue con valor por defecto 0.05 
    with_individual_plot (bool): Si es True, genera gráficos individuales para cada categoría.

    Retorna:
    Una lista de las columnas categóricas cuyo test de relación con el target es significativo.
    """
    
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Dataframe no válido.")
        return None
    if not isinstance(target_col, str) or target_col == "" or target_col not in df.columns:
        print("target_col no válido o no existe en el DataFrame.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("'target_col' debe ser una columna numérica.")
        return None
    if not isinstance(columns, list):
        print("'columns' debe ser una lista de strings.")
        return None
    if not isinstance(pvalue, (float, int)) or not (0 < pvalue < 1):
        print("'pvalue' debe ser un número entre 0 y 1.")
        return None
    if not isinstance(with_individual_plot, bool):
        print("'with_individual_plot' debe ser booleano.")
        return None

    # Si columns está vacía, usar columnas categóricas (excluyendo target_col)
    if not columns:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        columns = [col for col in columns if col != target_col]
    else:
        columns = [col for col in columns if col in df.columns and col != target_col]
        if not columns:
            print("No hay columnas válidas en 'columns'.")
            return None

    selected_cols = []
    for col in columns:
        # Comprobar que la columna es categórica
        if not (df[col].dtype == "object" or str(df[col].dtype).startswith("category")):
            continue

        temp_df = df[[col, target_col]].dropna() #eliminamos nulos para evitar problemas    

        categorias = df[col].dropna().unique()
        num_categorias = len(categorias)

        if num_categorias <2: #Checkeamos que hayan suficientes categorias
            continue

        grupos = [temp_df[temp_df[col] == cat][target_col].values
                  for cat in categorias]
        grupos = [g for g in grupos if len(g) >= 2]

        if len(grupos) < 2:
            continue

        try:
            if  num_categorias == 2:                
                estadistica, p_val1 = stats.ttest_ind(grupos[0], grupos[1])
                #estadistica, p_val2 = stats.mannwhitneyu(grupos[0], grupos[1], alternative= "two-sided")
            else:
                estadistica, p_val1 = stats.f_oneway(*grupos) 
                #estadistica, p_val2 = stats.kruskal(*grupos)

            if p_val1 < pvalue:# or p_val2 < pvalue:

                selected_cols.append(col)

                plt.figure(figsize=(8,4))
                sns.histplot(data=df, x=target_col, hue=col, element="step", stat="density", common_norm=False)
                plt.title(f"Histograma de {target_col} por {col} (p={p_val1:.3g})")
                plt.show()

                if with_individual_plot:
                    for cat in categorias:
                        plt.figure(figsize=(6,3))
                        sns.histplot(df[df[col]==cat][target_col], kde=True)
                        plt.title(f"{target_col} para {col} = {cat}")
                        plt.show()
                
        except Exception as e:
            print(f"Error procesado columna {col}; {e}")
            continue

    if not selected_cols:
        print("No se encontraron columnas categóricas relacionadas con el target para el nivel de significación indicado.")
    return selected_cols



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