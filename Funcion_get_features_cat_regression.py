"""
Esta funcion devuelve columnas categoricas con relacion significativa respecto al target

Variables: 

dataframe
target_col = Nombre de la columna target
pvalue = Nivel de significancia (segun el ejercicio, su valor por defecto sera 0.5)
usar_metrica: Para no hardcodearlo, me he permitido la libertad de añadir un parametro mas a la función

Devuelve una lista de las columnas con relacion significativa
"""

def get_features_cat_regression(dataframe, target_col, pvalue, usar_metrica = False):

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
            print(f"Error procesado columna ´{columna}; {e}")
            continue
    return columnas_significativas






resultado = get_features_cat_regression(df, "fare", 0.05)

print(f"Columnas significativas: {resultado}")