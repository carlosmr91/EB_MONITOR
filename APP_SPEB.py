# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:03:56 2024

@author: juan.melendez
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pyproj

# CONFIGURACIÓN DE LA PAGINA STREAMLIT
st.set_page_config(page_title="MONITOREO PRODUCCIÓN")
st.markdown("<h1 style='text-align: center; color: black;'>HISTÓRICO DE PRODUCCIÓN</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: green;'>TABLERO DE PRODUCCIÓN DIARIA Y ACUMULADA</h3>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color: gray;'>⚙️ CONTROLADORES</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'>________________________________</p>", unsafe_allow_html=True)
#st.sidebar.markdown("<h2 style='color: gray;'>ARCHIVO CSV UTF-8</h2>", unsafe_allow_html=True)

# CARGA DE ARCHIVO
uploaded_file = st.sidebar.file_uploader("📂 Cargar archivo Excel (formato CSV UTF-8)", type=["csv","CSV","TXT","txt"])
if uploaded_file is not None:
    # Leer el archivo Excel
    df = pd.read_csv(uploaded_file,sep=",")
    #st.dataframe(df)
else:
    st.toast("ARCHIVO NO CARGADO ❗❗")
    st.stop()

st.sidebar.markdown("<p style='text-align: center;'>________________________________</p>", unsafe_allow_html=True)            
alocada_df = pd.DataFrame(df)
            
pozoID = alocada_df["Pozo_Oficial"].unique()
boton_pozoID = st.sidebar.multiselect("Pozos", pozoID)
df_filtro = df[ df["Pozo_Oficial"].isin( boton_pozoID ) ]
df_filtro["Fecha"] = pd.to_datetime(df_filtro["Fecha"],format='%d/%m/%Y %H:%M')

max_numero_meses = df_filtro.groupby("Pozo_Oficial")["NumeroMeses"].max().reset_index()
df_max = pd.merge(df_filtro, max_numero_meses, on=["Pozo_Oficial", "NumeroMeses"])
df_max = df_max[["Pozo_Oficial", "NumeroMeses", "AceiteAcumulado Mbbl", "AguaAcumulada Mbbl","GasAcumulado MMpc"]]
#st.write(df_max)

# CALCULO DE NUEVAS COLUMNAS
df_filtro['1/qo'] = 1 / df_filtro['AceiteDiario bpd']
df_filtro['Np/qo'] = (df_filtro['AceiteAcumulado Mbbl'] * 1000) / df_filtro['AceiteDiario bpd']
df_filtro['Qw/Nb'] = df_filtro['AguaDiaria bpd'] / df_filtro['BrutoDiario bpd']

columns_to_group = {
    "BrutoDiario bpd":"SumaBruta_bpd",
    "AceiteDiario bpd": "SumaAceite_bpd",
    "AguaDiaria bpd": "SumaAgua_bpd",
    "GasDiario pcd": "SumaGas_pcd",
    "BrutoAcumulado Mbbl":"SumaBrutaAc_Mbbl",
    "AceiteAcumulado Mbbl": "SumaAceiteAc_Mbbl",
    "AguaAcumulada Mbbl": "SumaAguaAc_Mbbl",
    "GasAcumulado MMpc": "SumaGasAc_MMpc"
}

# AGRUPAMIENTO Y RENOMBRE DE COLUMNAS
for col, new_col in columns_to_group.items():
    grouped_df = df_filtro.groupby("Fecha")[col].sum().reset_index().rename(columns={col: new_col})
    df_filtro = pd.merge(df_filtro, grouped_df, on='Fecha', how='left')
#    df_filtro[new_col] = df_filtro[new_col].ffill()

#-------------------------------------------------------------
#-----------------SECCION DE GRAFICOS-------------------------
#-------------------------------------------------------------    

# Crear la figura del gráfico con subplots
plot_RPM = make_subplots(specs=[[{"secondary_y": False}]])

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any():
    # Ordenar el DataFrame por 'Fecha' y eliminar NaT
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')
    
    # Seleccionar datos para cada pozo y añadirlos al gráfico
    for pozo in boton_pozoID:
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]
        
        # Verificar si 'Fecha' tiene datos válidos
        if not df_pozo.empty and df_pozo['Fecha'].notna().any():
            plot_RPM.add_trace(
                go.Scatter(
                    x=df_pozo['Fecha'],
                    y=df_pozo['RPM'],
                    mode='lines+markers',
                    name=pozo,
                    marker=dict(symbol='cross', size=5)  # Ajustar el símbolo del marcador
                ),
                secondary_y=False
            )

    # Calcular el valor máximo del eje y y añadir un margen
    max_y = df_filtro['RPM'].max() * 1.1

    # Diseño del gráfico
    plot_RPM.update_layout(
        title="RPM",
        width=800,
        height=250,
        paper_bgcolor="#ECECEC",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(
            range=[0, max_y],
            title="RPM",
            side='left',
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='LightGray',
            zerolinewidth=1
        ),
        xaxis=dict(
            title="Fecha",
            showgrid=True,
            gridcolor='LightGray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='Black',
            zerolinewidth=1
        ),
        legend=dict(
            font=dict(
                size=15,
                family="Calibri",
                color="black",
            )
        )
    )

    # Diseño del título del eje y
    plot_RPM.update_yaxes(title_text="RPM", secondary_y=False)

# GRAFICO BRUTA DIARIO 
plot_BrutaDiaria = make_subplots(specs=[[{"secondary_y": False}]])

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any():
    # Asegurarse de que el DataFrame esté ordenado por fecha y sin valores NaT en 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # SELECCIONAR DATOS PARA CADA POZO
    for pozo in boton_pozoID:
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]

        # Verificar si 'Fecha' tiene información y no contiene valores NaT
        if df_pozo['Fecha'].notna().any():
            plot_BrutaDiaria.add_trace(
                go.Scatter(x=df_pozo['Fecha'], y=df_pozo['BrutoDiario bpd'], mode='lines', name=pozo),
                secondary_y=False,
            )

    # SELECCIONAR DATOS PARA CADA POZO (SUMA PERFIL)
    plot_BrutaDiaria.add_trace(
        go.Scatter(x=df_filtro['Fecha'], y=df_filtro['SumaBruta_bpd'], mode='markers', name='Suma Bruta',
                   marker=dict(color="black", symbol='cross', size=4)),
        secondary_y=False,
    )

    # CALCULAR EL MAXIMO DEL EJE Y
    max_y = df_filtro[['BrutoDiario bpd', 'SumaBruta_bpd']].max().max() * 1.1

    # DISEÑO DE GRAFICO
    plot_BrutaDiaria.update_layout(
        title="PRODUCCIÓN BRUTA",
        width=800,
        height=250,
        paper_bgcolor="#ECECEC",
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title="Bruta (bpd)", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
        legend=dict(
            font=dict(
                size=15,  # Aquí puedes cambiar el tamaño de la fuente de la leyenda
                family="Calibri",  # Familia de fuentes que soporta negritas
                color="black",
            )
        )
    )

    # DISEÑO TITULO DE EJES
    plot_BrutaDiaria.update_yaxes(title_text="Bruta (bpd)", secondary_y=False)
else:
    print("No hay datos disponibles en la columna 'Fecha'.")

# GRAFICO ACEITE DIARIO 
plot_AceiteDiario = make_subplots(specs=[[{"secondary_y": False}]])

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any():
    # Asegurarse de que el DataFrame esté ordenado por fecha y sin valores NaT en 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # SELECCIONAR DATOS PARA CADA POZO
    for pozo in boton_pozoID:
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]

        # Verificar si 'Fecha' tiene información y no contiene valores NaT
        if df_pozo['Fecha'].notna().any():
            plot_AceiteDiario.add_trace(
                go.Scatter(x=df_pozo['Fecha'], y=df_pozo['AceiteDiario bpd'], mode='lines', name=pozo),
                secondary_y=False,
            )

    # SELECCIONAR DATOS PARA CADA POZO (SUMA PERFIL)
    plot_AceiteDiario.add_trace(
        go.Scatter(x=df_filtro['Fecha'], y=df_filtro['SumaAceite_bpd'], mode='markers', name='Suma Neta',
                   marker=dict(color="black", symbol='cross', size=4)),
        secondary_y=False,
    )

# CALCULAR EL MAXIMO DEL EJE Y
max_y = df_filtro[['AceiteDiario bpd', 'SumaAceite_bpd']].max().max()* 1.1

# DISEÑO DE GRAFICO
plot_AceiteDiario.update_layout(
    title="PRODUCCIÓN NETA",
    width=800,
    height=250,
    paper_bgcolor="#E5FDDF",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(range=[0, max_y], title="Aceite (bpd)", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
    legend=dict(
        font=dict(
            size=15,  # Aquí puedes cambiar el tamaño de la fuente de la leyenda
            family="Calibri",  # Familia de fuentes que soporta negritas
            color="black",
        )
    )
)

# DISEÑO TITULO DE EJES
plot_AceiteDiario.update_yaxes(title_text="Aceite (bpd)", secondary_y=False)


# GRAFICO GAS DIARIO 
plot_GasDiario = make_subplots(specs=[[{"secondary_y": False}]])

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any():
    # Asegurarse de que el DataFrame esté ordenado por fecha y sin valores NaT en 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # SELECCIONAR DATOS PARA CADA POZO
    for pozo in boton_pozoID:
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]

        # Verificar si 'Fecha' tiene información y no contiene valores NaT
        if df_pozo['Fecha'].notna().any():
            plot_GasDiario.add_trace(
                go.Scatter(x=df_pozo['Fecha'], y=df_pozo['GasDiario pcd'], mode='lines', name=pozo),
                secondary_y=False,
            )

    # SELECCIONAR DATOS PARA CADA POZO (SUMA PERFIL)
    plot_GasDiario.add_trace(
        go.Scatter(x=df_filtro['Fecha'], y=df_filtro['SumaGas_pcd'], mode='markers', name='Suma Gas',
                   marker=dict(color="black", symbol='cross', size=4)),
        secondary_y=False,
    )

# CALCULAR EL MAXIMO DEL EJE Y
max_y = df_filtro[['GasDiario pcd', 'SumaGas_pcd']].max().max()* 1.1

# DISEÑO DE GRAFICO
plot_GasDiario.update_layout(
    title="PRODUCCIÓN DE GAS",
    width=800,
    height=250,
    paper_bgcolor="#FEEDE8",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(range=[0, max_y], title="Gas (pcd)", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
    legend=dict(
        font=dict(
            size=15,  # Aquí puedes cambiar el tamaño de la fuente de la leyenda
            family="Calibri",  # Familia de fuentes que soporta negritas
            color="black",
        )
    )
)

# DISEÑO TITULO DE EJES
plot_GasDiario.update_yaxes(title_text="Gas (pcd)", secondary_y=False)

# # GRAFICO AGUA DIARIA 
plot_AguaDiaria = make_subplots(specs=[[{"secondary_y": False}]])

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any():
    # Asegurarse de que el DataFrame esté ordenado por fecha y sin valores NaT en 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # SELECCIONAR DATOS PARA CADA POZO
    for pozo in boton_pozoID:
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]

        # Verificar si 'Fecha' tiene información y no contiene valores NaT
        if df_pozo['Fecha'].notna().any():
            plot_AguaDiaria.add_trace(
                go.Scatter(x=df_pozo['Fecha'], y=df_pozo['AguaDiaria bpd'], mode='lines', name=pozo),
                secondary_y=False,
            )

    # SELECCIONAR DATOS PARA CADA POZO (SUMA PERFIL)
    plot_AguaDiaria.add_trace(
        go.Scatter(x=df_filtro['Fecha'], y=df_filtro['SumaAgua_bpd'], mode='markers', name='Suma Agua',
                   marker=dict(color="black", symbol='cross', size=4)),
        secondary_y=False,
    )

# CALCULAR EL MAXIMO DEL EJE Y
max_y = df_filtro[['AguaDiaria bpd', 'SumaAgua_bpd']].max().max()* 1.1

# DISEÑO DE GRAFICO
plot_AguaDiaria.update_layout(
    title="PRODUCCIÓN DE AGUA",
    width=800,
    height=250,
    paper_bgcolor="#DFF9FD",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(range=[0, max_y], title="Agua (bpd)", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    )
)

# DISEÑO TITULO DE EJES
plot_AguaDiaria.update_yaxes(title_text="Agua (bpd)", secondary_y=False)

#-------------------------------------------------------------
#-----------------SECCION DE GRAFICOS ACUMULADA-------------------------
#-------------------------------------------------------------    

#  GRAFICO BRUTA ACUMULADA
# Obtener la fecha máxima del DataFrame
max_date = df_filtro['Fecha'].max()
# Inicializar la figura
plot_BrutaAc = make_subplots(specs=[[{"secondary_y": False}]])
# Inicializar carry forward
carry_forward = 0

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any() and pd.notna(max_date):
    # Ordenar el DataFrame por 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # Iterar sobre cada pozo único
    for pozo in boton_pozoID:
        # Filtrar datos para el pozo actual
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo].copy()
        
        # Verificar si el pozo tiene datos y fechas válidas
        if df_pozo['Fecha'].notna().any() and pd.notna(df_pozo['Fecha'].min()):
            # Obtener el valor máximo de 'BrutoAcumulado Mbbl' para el pozo actual
            max_value = df_pozo['BrutoAcumulado Mbbl'].max() * 1.1
            
            # Crear un DataFrame extendido hasta la fecha más reciente
            extended_dates = pd.date_range(start=df_pozo['Fecha'].min(), end=max_date, freq='D')
            extended_df = pd.DataFrame({'Fecha': extended_dates})
            extended_df = pd.merge(extended_df, df_pozo[['Fecha', 'BrutoAcumulado Mbbl']], on='Fecha', how='left')
            extended_df['BrutoAcumulado Mbbl'].fillna(method='ffill', inplace=True)
            extended_df['BrutoAcumulado Mbbl'].fillna(max_value, inplace=True)
            
            # Agregar carry forward al DataFrame extendido
            #extended_df['BrutoAcumulado Mbbl'] += carry_forward
            
            # Actualizar carry forward para el siguiente pozo
            carry_forward = extended_df['BrutoAcumulado Mbbl'].iloc[-1]
            
            # Añadir la traza al gráfico
            plot_BrutaAc.add_trace(
                go.Scatter(
                    x=extended_df['Fecha'],
                    y=extended_df['BrutoAcumulado Mbbl'],
                    hoverinfo='x+y',
                    mode='none',
                    name=pozo,
                    fill='tonexty',
                    stackgroup='one'
                ),
            )

# Diseño del gráfico
plot_BrutaAc.update_layout(
    title="PRODUCCIÓN BRUTA ACUMULADA",
    width=800,
    height=250,
    paper_bgcolor="#ECECEC",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(title="Bruta Acumulada (Mbbl)", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    )
)

# DISEÑO TITULO DE EJES
plot_BrutaAc.update_yaxes(title_text="Bruta Acumulada (Mbbl)", secondary_y=False)

# # GRAFICO NETA ACUMULADA
# Inicializar la figura
plot_NetaAc = make_subplots(specs=[[{"secondary_y": False}]])

# Inicializar carry forward
carry_forward = 0

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any() and pd.notna(max_date):
    # Ordenar el DataFrame por 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # Iterar sobre cada pozo único
    for pozo in boton_pozoID:
        # Filtrar datos para el pozo actual
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo].copy()
        
        # Verificar si el pozo tiene datos y fechas válidas
        if df_pozo['Fecha'].notna().any() and pd.notna(df_pozo['Fecha'].min()):
            # Obtener el valor máximo de 'BrutoAcumulado Mbbl' para el pozo actual
            max_value = df_pozo['AceiteAcumulado Mbbl'].max() * 1.1
            
            # Crear un DataFrame extendido hasta la fecha más reciente
            extended_dates = pd.date_range(start=df_pozo['Fecha'].min(), end=max_date, freq='D')
            extended_df = pd.DataFrame({'Fecha': extended_dates})
            extended_df = pd.merge(extended_df, df_pozo[['Fecha', 'AceiteAcumulado Mbbl']], on='Fecha', how='left')
            extended_df['AceiteAcumulado Mbbl'].fillna(method='ffill', inplace=True)
            extended_df['AceiteAcumulado Mbbl'].fillna(max_value, inplace=True)
            
            # Agregar carry forward al DataFrame extendido
            #extended_df['AceiteAcumulado Mbbl'] += carry_forward
            
            # Actualizar carry forward para el siguiente pozo
            carry_forward = extended_df['AceiteAcumulado Mbbl'].iloc[-1]
            
            # Añadir la traza al gráfico
            plot_NetaAc.add_trace(
                go.Scatter(
                    x=extended_df['Fecha'],
                    y=extended_df['AceiteAcumulado Mbbl'],
                    hoverinfo='x+y',
                    mode='none',
                    name=pozo,
                    fill='tonexty',
                    stackgroup='one'
                ),
            )

# Diseño del gráfico
plot_NetaAc.update_layout(
    title="PRODUCCIÓN NETA ACUMULADA",
    width=800,
    height=250,
    paper_bgcolor="#E5FDDF",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(title="Neta Acumulada (Mbbl)", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    )
)

# DISEÑO TITULO DE EJES
plot_NetaAc.update_yaxes(title_text="Neta Acumulada (Mbbl)", secondary_y=False)

# # GRAFICO GAS ACUMULADA
# # Inicializar la figura
plot_GasAc = make_subplots(specs=[[{"secondary_y": False}]])

# Inicializar carry forward
carry_forward = 0

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any() and pd.notna(max_date):
    # Ordenar el DataFrame por 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # Iterar sobre cada pozo único
    for pozo in boton_pozoID:
        # Filtrar datos para el pozo actual
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo].copy()
        
        # Verificar si el pozo tiene datos y fechas válidas
        if df_pozo['Fecha'].notna().any() and pd.notna(df_pozo['Fecha'].min()):
            # Obtener el valor máximo de 'BrutoAcumulado Mbbl' para el pozo actual
            max_value = df_pozo['GasAcumulado MMpc'].max() * 1.1
            
            # Crear un DataFrame extendido hasta la fecha más reciente
            extended_dates = pd.date_range(start=df_pozo['Fecha'].min(), end=max_date, freq='D')
            extended_df = pd.DataFrame({'Fecha': extended_dates})
            extended_df = pd.merge(extended_df, df_pozo[['Fecha', 'GasAcumulado MMpc']], on='Fecha', how='left')
            extended_df['GasAcumulado MMpc'].fillna(method='ffill', inplace=True)
            extended_df['GasAcumulado MMpc'].fillna(max_value, inplace=True)
            
            # Agregar carry forward al DataFrame extendido
            #extended_df['AceiteAcumulado Mbbl'] += carry_forward
            
            # Actualizar carry forward para el siguiente pozo
            carry_forward = extended_df['GasAcumulado MMpc'].iloc[-1]
            
            # Añadir la traza al gráfico
            plot_GasAc.add_trace(
                go.Scatter(
                    x=extended_df['Fecha'],
                    y=extended_df['GasAcumulado MMpc'],
                    hoverinfo='x+y',
                    mode='none',
                    name=pozo,
                    fill='tonexty',
                    stackgroup='one'
                ),
            )

# Diseño del gráfico
plot_GasAc.update_layout(
    title="PRODUCCIÓN ACUMULADA DE GAS",
    width=800,
    height=250,
    paper_bgcolor="#FEEDE8",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(title="Gas Acumulado (Mbbl)", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    )
)

# DISEÑO TITULO DE EJES
plot_GasAc.update_yaxes(title_text="Gas Acumulado (MMpc)", secondary_y=False)

# # GRAFICO AGUA ACUMULADA
# Inicializar la figura
plot_AguaAc = make_subplots(specs=[[{"secondary_y": False}]])

# Inicializar carry forward
carry_forward = 0

# Verificar si 'Fecha' tiene información y no contiene valores NaT
if df_filtro['Fecha'].notna().any() and pd.notna(max_date):
    # Ordenar el DataFrame por 'Fecha'
    df_filtro = df_filtro.dropna(subset=['Fecha']).sort_values(by='Fecha')

    # Iterar sobre cada pozo único
    for pozo in boton_pozoID:
        # Filtrar datos para el pozo actual
        df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo].copy()
        
        # Verificar si el pozo tiene datos y fechas válidas
        if df_pozo['Fecha'].notna().any() and pd.notna(df_pozo['Fecha'].min()):
            # Obtener el valor máximo de 'BrutoAcumulado Mbbl' para el pozo actual
            max_value = df_pozo['AguaAcumulada Mbbl'].max() * 1.1
            
            # Crear un DataFrame extendido hasta la fecha más reciente
            extended_dates = pd.date_range(start=df_pozo['Fecha'].min(), end=max_date, freq='D')
            extended_df = pd.DataFrame({'Fecha': extended_dates})
            extended_df = pd.merge(extended_df, df_pozo[['Fecha', 'AguaAcumulada Mbbl']], on='Fecha', how='left')
            extended_df['AguaAcumulada Mbbl'].fillna(method='ffill', inplace=True)
            extended_df['AguaAcumulada Mbbl'].fillna(max_value, inplace=True)
            
            # Agregar carry forward al DataFrame extendido
            #extended_df['AceiteAcumulado Mbbl'] += carry_forward
            
            # Actualizar carry forward para el siguiente pozo
            carry_forward = extended_df['AguaAcumulada Mbbl'].iloc[-1]
            
            # Añadir la traza al gráfico
            plot_AguaAc.add_trace(
                go.Scatter(
                    x=extended_df['Fecha'],
                    y=extended_df['AguaAcumulada Mbbl'],
                    hoverinfo='x+y',
                    mode='none',
                    name=pozo,
                    fill='tonexty',
                    stackgroup='one'
                ),
            )

# Diseño del gráfico
plot_AguaAc.update_layout(
    title="PRODUCCIÓN ACUMULADA DE AGUA",
    width=800,
    height=250,
    paper_bgcolor="#DFF9FD",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(title="Agua Acumulada Mbbl", side='left', showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1),
    xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1),
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    )
)

# DISEÑO TITULO DE EJES
plot_AguaAc.update_yaxes(title_text="Agua Acumulada (Mbbl)", secondary_y=False)

#_________________________________________________________________________________

plot_QoNp = make_subplots(specs=[[{"secondary_y": True}]])

#-----Agregar traza para AceiteAcumulado_bpd
for pozo in df_filtro['Pozo_Oficial'].unique():
    df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]
    plot_QoNp.add_trace(
        go.Scatter(
            x=df_pozo['AceiteAcumulado Mbbl'], 
            y=df_pozo['AceiteDiario bpd'], 
            mode='lines+markers', 
            name=pozo,
                    line=dict(dash='dot', width=2),
                    marker=dict(size=8, opacity=0.8)),
        secondary_y=False,
    ) 

plot_QoNp.update_layout(
    title="Relación Aceite Acumulado & Producción Neta Diaria",
    width=650,
    height=400,
    paper_bgcolor="#E5FDDF",
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis=dict(
        range=[-1, 3],  # Establecer rango para ambos ejes y
        dtick=1,  # Configura los ticks para una cuadrícula logarítmica adecuada
        title="Relación Aceite Acumulado & Producción Neta Diaria",
        side='left',showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
        type="log",
        ), 
    legend=dict(
        font=dict(
        size=15,
        family="Calibri",
        color="black",
        )
    ),
)

#------Actualizar títulos de los ejes
plot_QoNp.update_yaxes(title_text="Aceite (bpd)", secondary_y=False)
plot_QoNp.update_xaxes(title_text="Aceite Acumulado (Mbbl)")

plot_QwNp = make_subplots(specs=[[{"secondary_y": True}]])

#-----Agregar traza para AceiteAcumulado_bpd
for pozo in df_filtro['Pozo_Oficial'].unique():
    df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]
    plot_QwNp.add_trace(
        go.Scatter(
            x=df_pozo['AceiteAcumulado Mbbl'], 
            y=df_pozo['AguaDiaria bpd'], 
            mode='lines+markers', 
            name=pozo,
                    line=dict(dash='dot', width=2),
                    marker=dict(size=8, opacity=0.8)),
        secondary_y=False,
    ) 

plot_QwNp.update_layout(
    title="Relación Aceite Acumulado & Producción Agua Diaria",
    width=650,
    height=400,
    paper_bgcolor="#DFF9FD",
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis=dict(
        range=[-1, 3],  # Establecer rango para ambos ejes y
        dtick=1,  # Configura los ticks para una cuadrícula logarítmica adecuada
        title="Relación Aceite Acumulado & Producción Agua Diaria",
        side='left',showgrid=True, gridcolor='LightGray', gridwidth=1, zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
        type="log",
        ), 
    legend=dict(
        font=dict(
        size=15,
        family="Calibri",
        color="black",
        )
    ),
)

#------Actualizar títulos de los ejes
plot_QwNp.update_yaxes(title_text="Agua (bpd)", secondary_y=False)
plot_QwNp.update_xaxes(title_text="Aceite Acumulado (Mbbl)")



plot_QgNp = make_subplots(specs=[[{"secondary_y": True}]])

#-----Agregar traza para AceiteAcumulado_bpd
for pozo in df_filtro['Pozo_Oficial'].unique():
    df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]
    plot_QgNp.add_trace(
        go.Scatter(
            x=df_pozo['AceiteAcumulado Mbbl'], 
            y=df_pozo['GasDiario pcd'], 
            mode='lines+markers', 
            name=pozo,
                    line=dict(dash='dot', width=2),
                    marker=dict(size=8, opacity=0.8)),
        secondary_y=False,
    ) 

plot_QgNp.update_layout(
    title="Relación Aceite Acumulado & Producción Gas Diaria",
    width=650,
    height=400,
    paper_bgcolor="#FEEDE8",
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis=dict(
        range=[0, df_pozo['GasDiario pcd']],  # Establecer rango para ambos ejes y
        title="GasDiario pcd",
        side='left',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='LightGray',
        zerolinewidth=1,
        type="linear",  # Escala lineal
    ), 
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    ),
)

#------Actualizar títulos de los ejes
plot_QgNp.update_yaxes(title_text="Gas Diario (pcd)", secondary_y=False)
plot_QgNp.update_xaxes(title_text="Aceite Acumulado (Mbbl)")

# GRAFICO MESES-NP
plot_TNp = make_subplots(specs=[[{"secondary_y": True}]])

#-----Agregar traza para AceiteAcumulado_bpd
for pozo in df_filtro['Pozo_Oficial'].unique():
    df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]
    plot_TNp.add_trace(
        go.Scatter(
            x=df_pozo['NumeroMeses'], 
            y=df_pozo['AceiteAcumulado Mbbl'], 
            mode='lines+markers', 
            name=pozo,
                    line=dict(dash='dot', width=2),
                    marker=dict(size=8, opacity=0.8)),
        secondary_y=False,
    ) 

plot_TNp.update_layout(
    title="Relación Tiempo & Aceite Acumulado",
    width=650,
    height=400,
    paper_bgcolor="#E5FDDF",
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis=dict(
        range=[0, df_pozo['AceiteAcumulado Mbbl']],  # Establecer rango para ambos ejes y
        title="Aceite Acumulado (Mbbl)",
        side='left',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='LightGray',
        zerolinewidth=1,
        type="linear",  # Escala lineal
    ), 
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    ),
)

#------Actualizar títulos de los ejes
plot_TNp.update_yaxes(title_text="Aceite Acumulado (Mbbl)", secondary_y=False)
plot_TNp.update_xaxes(title_text="Meses")


# GRAFICO MESES-WP
plot_TWp = make_subplots(specs=[[{"secondary_y": True}]])

#-----Agregar traza para AceiteAcumulado_bpd
for pozo in df_filtro['Pozo_Oficial'].unique():
    df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]
    plot_TWp.add_trace(
        go.Scatter(
            x=df_pozo['NumeroMeses'], 
            y=df_pozo['AguaAcumulada Mbbl'], 
            mode='lines+markers', 
            name=pozo,
                    line=dict(dash='dot', width=2),
                    marker=dict(size=8, opacity=0.8)),
        secondary_y=False,
    ) 

plot_TWp.update_layout(
    title="Relación Tiempo & Agua Acumulada",
    width=650,
    height=400,
    paper_bgcolor="#DFF9FD",
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis=dict(
        range=[0, df_pozo['AguaAcumulada Mbbl']],  # Establecer rango para ambos ejes y
        title="Agua Acumulada (Mbbl)",
        side='left',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='LightGray',
        zerolinewidth=1,
        type="linear",  # Escala lineal
    ), 
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    ),
)

#------Actualizar títulos de los ejes
plot_TWp.update_yaxes(title_text="Agua Acumulada (Mbbl)", secondary_y=False)
plot_TWp.update_xaxes(title_text="Meses")


# GRAFICO MESES-GP
plot_TGp = make_subplots(specs=[[{"secondary_y": True}]])

#-----Agregar traza para AceiteAcumulado_bpd
for pozo in df_filtro['Pozo_Oficial'].unique():
    df_pozo = df_filtro[df_filtro['Pozo_Oficial'] == pozo]
    plot_TGp.add_trace(
        go.Scatter(
            x=df_pozo['NumeroMeses'], 
            y=df_pozo['GasAcumulado MMpc'], 
            mode='lines+markers', 
            name=pozo,
                    line=dict(dash='dot', width=2),
                    marker=dict(size=8, opacity=0.8)),
        secondary_y=False,
    ) 

plot_TGp.update_layout(
    title="Relación Tiempo & Gas Acumulado",
    width=650,
    height=400,
    paper_bgcolor="#FEEDE8",
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis=dict(
        range=[0, df_pozo['GasAcumulado MMpc']],  # Establecer rango para ambos ejes y
        title="Gas Acumulado (MMpc)",
        side='left',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='LightGray',
        zerolinewidth=1,
        type="linear",  # Escala lineal
    ), 
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    ),
)

#------Actualizar títulos de los ejes
plot_TGp.update_yaxes(title_text="Gas Acumulado (MMpc)", secondary_y=False)
plot_TGp.update_xaxes(title_text="Meses")

# CREAR GRAFICO
fig = go.Figure()

# Agregar una traza para cada pozo
for pozo in df_max["Pozo_Oficial"].unique():
    df_pozo = df_max[df_max["Pozo_Oficial"] == pozo]
    fig.add_trace(go.Scatter(
        x=df_pozo["AguaAcumulada Mbbl"], 
        y=df_pozo["AceiteAcumulado Mbbl"], 
        mode='markers+text',  # Añade etiquetas de texto a los puntos
        text=df_pozo["Pozo_Oficial"],  # Etiqueta con el nombre del pozo
        textposition='top center',  # Posición del texto
        marker=dict(
            size=10, 
            line=dict(width=1, color='black')
        ),
        name=pozo  # Nombre de la traza, que aparecerá en la leyenda
    ))

# Configuración del layout del gráfico
fig.update_layout(
    title="Relación Agua Acumulada & Aceite Acumulado",
    width=750,
    height=400,
    xaxis_title="Agua Acumulada (Mbbl)",
    paper_bgcolor='#DEDEDE',
    plot_bgcolor='rgb(255, 255, 255)',
    margin=dict(l=40, r=40, t=40, b=40),
    yaxis=dict(
        type="log",  # Escala logarítmica
        title="Aceite Acumulado (Mbbl)",
        side='left',
        showgrid=True,
        gridcolor='LightGray',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='LightGray',
        zerolinewidth=1,
        range=[-1, 3],  # Rango de 0.1 a 1000 en escala logarítmica
        dtick=1  # Configura los ticks para una cuadrícula logarítmica adecuada
    ),
    legend=dict(
        font=dict(
            size=15,
            family="Calibri",
            color="black",
        )
    )
)


# ESTRUCTURA DE PAGINA
tabs = st.tabs([" PRODUCCIÓN ALOCADA ", " TENDENCIAS ACUMULADA ", "DINAMICO"])
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart( plot_BrutaDiaria )
        st.plotly_chart( plot_AceiteDiario )
        st.plotly_chart( plot_AguaDiaria )
        st.plotly_chart( plot_GasDiario )
        st.plotly_chart( plot_RPM )
    with c2:
        st.plotly_chart( plot_BrutaAc )
        st.plotly_chart( plot_NetaAc )
        st.plotly_chart( plot_AguaAc )
        st.plotly_chart( plot_GasAc )


with tabs[1]:
    c1,c2,c3 = st.columns(3)
    with c1:
        #st.plotly_chart( plot_RPM2 )
        st.plotly_chart(plot_TNp)
        st.plotly_chart(plot_QoNp)
    with c2:
        st.plotly_chart(plot_TWp)
        st.plotly_chart(plot_QwNp)
        #st.plotly_chart( fig )
    with c3:
        st.plotly_chart(plot_TGp )
        st.plotly_chart(plot_QgNp)

with tabs[2]:
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart( fig )
        st.write(df_max)
    # with c2:
        
    # with c3:
        
# Mostrar los DataFrames cargados
# st.write("Datos de Alocada:")
# st.write(df_filtro)