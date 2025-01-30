import numpy as np
import pandas as pd

def tipo_equipo(df: pd.DataFrame, equipo: str):
    equipo_df =  df[df['Equipo'] ==  equipo].copy()
    equipo_df = equipo_df.reset_index(drop=True)
    lista_tipos = equipo_df.Tipo.unique()
    return lista_tipos

def mdf_equipo(df: pd.DataFrame, equipo: str):
    equipo_df =  df[df['Equipo'] ==  equipo].copy()
    equipo_df = equipo_df.reset_index(drop=True)
    lista_mdf = equipo_df.mdf.unique()
    return lista_mdf

def tbf(df: pd.DataFrame, equipo_nombre: str, tipos: list):
    df_equipo = df[df['Equipo'] == equipo_nombre].copy()
    df_equipo.sort_values(by='Fecha_Inicio', inplace=True)
    df_equipo.reset_index(drop=True, inplace=True)

    df_equipo['Diferencia'] = (df_equipo['Fecha_Inicio'] - df_equipo['Fecha_Fin'].shift()).dt.total_seconds() / 3600
    df_equipo['Diferencia'].fillna(0, inplace=True)

    df_tipos = df_equipo[df_equipo['Tipo'].isin(tipos)].copy()

    indices = df_tipos.index.tolist()
    df_equipo['TBX'] = 0.0

    for i in range(len(indices) - 1):
        suma_intervalo = np.sum(df_equipo['Diferencia'][indices[i]+1:indices[i+1]+1])
        df_equipo.at[indices[i+1], 'TBX'] = suma_intervalo
        #print(indices[i]+1, suma_intervalo, indices[i+1]+1)

    #return df_equipo[['Equipo', 'Fecha_Inicio', 'Fecha_Fin', 'Tipo', 'mdf', 'Diferencia', 'TBX', 'TTX']]
    return df_equipo[['Equipo', 'Fecha_Inicio', 'Fecha_Fin', 'Tipo', 'mdf', 'TBX', 'TTX']]


def tratamiento(repo: pd.DataFrame):
    #repo = pd.read_csv("LS2 repo.csv", sep=';', decimal=",", encoding='latin-1')
    repo['Fecha_Inicio'] = pd.to_datetime(repo['Fecha'] + ' ' + repo['Hora'], infer_datetime_format=True, dayfirst=True)
    repo = repo[['Fecha_Inicio', 'Duracion', 'Tipo', 'Equipo', 'Modo de Falla']]
    repo.drop_duplicates(inplace=True)
    repo.sort_values(by='Fecha_Inicio', inplace=True)
    repo = repo.reset_index(drop=True)

    repo['Fecha_Fin'] = repo['Fecha_Inicio'] + pd.to_timedelta(repo['Duracion'], unit='h')
    repo = repo[['Fecha_Inicio', 'Fecha_Fin', 'Duracion', 'Tipo', 'Equipo', 'Modo de Falla']]
    repo = repo.rename(columns={'Modo de Falla': 'mdf'})
    repo = repo.rename(columns={'Duracion': 'TTX'})
    return repo

def calcular_valores(best_fit, tiempo, tipo):
    if tipo == 'TBX':
        confiabilidad = best_fit.SF(tiempo)
        tasa_falla = best_fit.HF(tiempo)
        return confiabilidad, tasa_falla, None, None
    elif tipo == 'TTX':
        densidad_probabilidad = best_fit.PDF(tiempo)
        probabilidad_acumulada = best_fit.CDF(tiempo)
        return None, None, densidad_probabilidad, probabilidad_acumulada