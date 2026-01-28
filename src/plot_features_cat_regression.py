import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' debe ser un DataFrame de pandas.")
        return None
    if not isinstance(target_col, str) or target_col == "" or target_col not in df.columns:
        print("Debe indicar una columna objetivo válida en 'target_col'.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("'target_col' debe ser una columna numérica.")
        return None
    if not isinstance(columns, list):
        print("El argumento 'columns' debe ser una lista de strings.")
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
        # ANOVA: comparar medias de target_col entre grupos de col
        grupos = [df[df[col] == cat][target_col].dropna() for cat in df[col].dropna().unique()]
        if len(grupos) < 2:
            continue
        try:
            stat, p = f_oneway(*grupos)
        except Exception:
            continue
        if p < pvalue:
            selected_cols.append(col)
            plt.figure(figsize=(8,4))
            sns.histplot(data=df, x=target_col, hue=col, element="step", stat="density", common_norm=False)
            plt.title(f"Histograma de {target_col} por {col} (p={p:.3g})")
            plt.show()
            if with_individual_plot:
                for cat in df[col].dropna().unique():
                    plt.figure(figsize=(6,3))
                    sns.histplot(df[df[col]==cat][target_col], kde=True)
                    plt.title(f"{target_col} para {col} = {cat}")
                    plt.show()
    if not selected_cols:
        print("No se encontraron columnas categóricas relacionadas con el target para el nivel de significación indicado.")
    return selected_cols
