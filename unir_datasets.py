import pandas as pd

# Cargar los datasets
df1 = pd.read_csv("dataset_indoor_etiquetado_reducido.csv")
df2 = pd.read_csv("dataset_indoor_etiquetado_reducido (1).csv")

# Verificar que tengan las mismas columnas
assert df1.columns.equals(df2.columns), "Las columnas no coinciden."

# Unir verticalmente
df_combinado = pd.concat([df1, df2], axis=0, ignore_index=True)

# Guardar el resultado
df_combinado.to_csv("dataset_combinado.csv", index=False)
