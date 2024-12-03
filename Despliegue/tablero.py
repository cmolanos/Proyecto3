import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

# Cargar los datos
file_path = 'Base_Limpia.xlsx'
df = pd.read_excel(file_path)

# Cargar los modelos preentrenados
model_class = load_model("modelo_clasificacion.h5")
model_reg = load_model("modelo_reg.h5")

# Cargar los scalers preentrenados
with open("objetos_adicionales_clasificacion.pkl", "rb") as f:
    scaler_class = pickle.load(f)

with open("objetos_adicionales_reg.pkl", "rb") as f:
    scaler_reg = pickle.load(f)

# Crear la app
app = Dash(__name__, suppress_callback_exceptions=True)

categorical_columns = ['cole_area_ubicacion', 'cole_caracter', 'cole_bilingue', 'cole_jornada',
    'fami_tienecomputador', 'fami_tieneinternet', 'fami_tienelavadora', 'fami_tieneautomovil','fami_estratovivienda','fami_personashogar']
numerical_columns = ['edad_examen']

# Layout de la app
app.layout = html.Div(style={'backgroundColor': '#f3f4f6', 'padding': '20px'}, children=[
    # Título del tablero
    html.H1("Análisis Resultados ICFES Departamento de Caldas",
            style={'text-align': 'center', 'margin-bottom': '20px', 'font-weight': 'bold', 'color': '#2c3e50'}),

    # Filtro de periodo
    html.Div([
        html.Label("Selecciona un periodo:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='filter-periodo',
            options=[{'label': periodo, 'value': periodo} for periodo in sorted(df['periodo'].unique())],
            placeholder="Selecciona un periodo",
            style={'margin-bottom': '10px'}
        ),
    ], style={'margin-bottom': '20px'}),

    # Métricas generales
        html.Div(id='metrics-output', style={
        'background-color': '#ffffff',
        'padding': '20px',
        'border-radius': '10px',
        'box-shadow': '0px 4px 6px rgba(0,0,0,0.1)',
        'margin-bottom': '20px',
        'font-size': '16px',
        'color': '#34495e'
    }),

    # Gráficas principales
    html.Div([
        # Gráfica 1: Análisis por Municipio
        html.Div([
            html.H3("Análisis por Municipio", style={'color': '#34495e'}),
            html.Label("Filtrar por municipio:"),
            dcc.Dropdown(
                id='filter-municipio',
                placeholder="Selecciona un municipio",
                style={'margin-bottom': '10px'}
            ),
            dcc.Graph(id='score-graph-municipio'),
        ], style={'margin-bottom': '20px', 'background-color': '#fff', 'border-radius': '10px', 
                  'padding': '20px', 'box-shadow': '0px 4px 6px rgba(0,0,0,0.1)'}),

        # Gráfica 2: Análisis de Desempeño
        html.Div([
            html.H3("Análisis de Desempeño", style={'color': '#34495e'}),
            dcc.RadioItems(
                id='analysis-type',
                options=[
                    {'label': 'Top 5 Mejor Desempeño', 'value': 'top_mejores'},
                    {'label': 'Top 5 Peor Desempeño', 'value': 'top_peores'}
                ],
                value='top_mejores',
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Dropdown(
                id='puntaje-type',
                options=[
                    {'label': 'Puntaje Global', 'value': 'punt_global'},
                    {'label': 'Puntaje de Inglés', 'value': 'punt_ingles'},
                    {'label': 'Puntaje de Sociales y Ciudadanas', 'value': 'punt_sociales_ciudadanas'},
                    {'label': 'Puntaje de Matemáticas', 'value': 'punt_matematicas'},
                    {'label': 'Puntaje de Ciencias Naturales', 'value': 'punt_c_naturales'}
                ],
                value='punt_global',
                placeholder="Selecciona un tipo de puntaje",
            ),
            dcc.Graph(id='performance-graph'),
        ], style={'margin-bottom': '20px', 'background-color': '#fff', 'border-radius': '10px', 
                  'padding': '20px', 'box-shadow': '0px 4px 6px rgba(0,0,0,0.1)'}),

        # Gráfica 3: Mapa de Calor
        html.Div([
            html.H3("Mapa de Calor: Distribución de Puntajes", style={'color': '#34495e'}),
            html.Label("Selecciona el eje X (variable categórica):"),
            dcc.Dropdown(
                id='heatmap-x-axis',
                options=[{'label': col, 'value': col} for col in df.select_dtypes(include='object').columns],
                placeholder="Selecciona una variable para el eje X",
            ),
            html.Label("Selecciona el eje Y (variable categórica):"),
            dcc.Dropdown(
                id='heatmap-y-axis',
                options=[{'label': col, 'value': col} for col in df.select_dtypes(include='object').columns],
                placeholder="Selecciona una variable para el eje Y",
            ),
             dcc.Graph(id='heatmap-graph'),
        ], style={'background-color': '#fff', 'border-radius': '10px', 
                  'padding': '20px', 'box-shadow': '0px 4px 6px rgba(0,0,0,0.1)'})
    ]),

    html.H1("Predicción de Puntajes ICFES", style={
        'text-align': 'center',
        'margin-bottom': '20px',
        'font-weight': 'bold',
        'color': '#2c3e50'
    }),

    # Selector de modelo
    html.Div([
        html.Label("Selecciona el tipo de predicción:", style={'font-weight': 'bold'}),
        dcc.RadioItems(
            id='model-selection',
            options=[
                {'label': 'Clasificación (probabilidad de puntaje > 300)', 'value': 'classification'},
                {'label': 'Regresión (predicción exacta del puntaje)', 'value': 'regression'}
            ],
            value='classification',
            labelStyle={'display': 'block'}
        ),
    ], style={'margin-bottom': '20px'}),

html.Div([
    html.H3("Características del estudiante", style={'color': '#34495e'}),
    html.Div([
        # Área de ubicación del colegio
        html.Label("Área de ubicación del colegio:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-cole_area_ubicacion',
            options=[
                {'label': 'Urbano', 'value': 1},
                {'label': 'Rural', 'value': 0}
            ],
            value=1,
            style={'margin-bottom': '10px'}
        ),

        # Carácter del colegio
        html.Label("Carácter del colegio:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-cole_caracter',
            options=[
                {'label': 'NO APLICA', 'value': 3},
                {'label': 'TÉCNICO', 'value': 2},
                {'label': 'ACADÉMICO', 'value': 0},
                {'label': 'TÉCNICO/ACADÉMICO', 'value': 1}
            ],
            value=1,
            style={'margin-bottom': '10px'}
        ),

        # Colegio bilingüe
        html.Label("¿Es el colegio bilingüe?", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-cole_bilingue',
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Sí', 'value': 1}
            ],
            value=1,
            style={'margin-bottom': '10px'}
        ),

        # Jornada del colegio
        html.Label("Jornada del colegio:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-cole_jornada',
            options=[
                {'label': 'COMPLETA', 'value': 1},
                {'label': 'MAÑANA', 'value': 0},
                {'label': 'TARDE', 'value': 5},
                {'label': 'ÚNICA', 'value': 2},
                {'label': 'NOCHE', 'value': 3},
                {'label': 'SABATINA', 'value': 4}
            ],
            value=1,
            style={'margin-bottom': '10px'}
        ),

        # Tiene computador
        html.Label("¿Tiene computador en casa?", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-fami_tienecomputador',
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Sí', 'value': 1}
            ],
            value=1,
            style={'margin-bottom': '10px'}
        ),

        # Tiene internet
        html.Label("¿Tiene internet en casa?", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-fami_tieneinternet',
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Sí', 'value': 1}
            ],
            value=1,
            style={'margin-bottom': '10px'}
        ),

        # Tiene lavadora
        html.Label("¿Tiene lavadora en casa?", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-fami_tienelavadora',
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Sí', 'value': 1}
            ],
            value=1,
            style={'margin-bottom': '10px'}
        ),

        # Tiene automóvil
        html.Label("¿Tiene automóvil en casa?", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-fami_tieneautomovil',
            options=[
                {'label': 'No', 'value': 0},
                {'label': 'Sí', 'value': 1}
            ],
            value=0,
            style={'margin-bottom': '10px'}
        ),

        # Estrato socioeconómico
        html.Label("Estrato socioeconómico:", style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='input-fami_estratovivienda',
            options=[
                {'label': 'Estrato 1', 'value': 1},
                {'label': 'Estrato 2', 'value': 2},
                {'label': 'Estrato 5', 'value': 5},
                {'label': 'Sin Estrato', 'value': 0},
                {'label': 'Estrato 3', 'value': 3},
                {'label': 'Estrato 6', 'value': 6},
                {'label': 'Estrato 4', 'value': 4}
            ],
            value=2,
            style={'margin-bottom': '10px'}
        ),

        # Personas en el hogar
        html.Label("Personas en el hogar:", style={'font-weight': 'bold'}),
        dcc.Input(
            id='input-fami_personashogar', 
            type='number', 
            min=0, 
            max=16, 
            value=3, 
            style={'margin-bottom': '10px'}
        ),

        # Edad del estudiante
        html.Label("Edad del estudiante:", style={'font-weight': 'bold'}),
        dcc.Input(
            id='input-edad_examen',
            type='number',
            min=10,
            max=100,
            value=18,
            style={'margin-bottom': '10px'}
        ),
    ], style={
        'padding': '20px', 
        'background-color': '#ffffff', 
        'border-radius': '10px', 
        'box-shadow': '0px 4px 6px rgba(0,0,0,0.1)', 
        'margin-bottom': '20px'
    }),

    html.Button('Calcular', id='calculate-button', n_clicks=0, style={
        'background-color': '#4CAF50',
        'color': 'white',
        'padding': '10px 20px',
        'border': 'none',
        'border-radius': '5px',
        'cursor': 'pointer',
        'font-size': '16px'
    }),

    html.Div(
        id='prediction-output', 
        style={'margin-top': '20px', 'font-size': '20px', 'color': '#34495e'}
    ),
    ])
])


@app.callback(
    [Output('metrics-output', 'children'),
     Output('filter-municipio', 'options')],
    Input('filter-periodo', 'value')
)
def update_metrics_and_municipios(periodo):
    # Filtrar los datos por periodo
    filtered_df = df if not periodo else df[df['periodo'] == periodo]

    # Si no hay datos, devolver mensaje y opciones vacías
    if filtered_df.empty:
        return html.Div("No hay datos disponibles para el periodo seleccionado.", style={'color': '#c0392b'}), []

    # Calcular métricas
    avg_scores_municipio = filtered_df.groupby('cole_mcpio_ubicacion')['punt_global'].mean()
    municipio_mejor = avg_scores_municipio.idxmax()
    puntaje_mejor_municipio = avg_scores_municipio.max()

    municipio_peor = avg_scores_municipio.idxmin()
    puntaje_peor_municipio = avg_scores_municipio.min()

    avg_scores_colegio = filtered_df.groupby('cole_nombre_establecimiento')['punt_global'].mean()
    colegio_mejor = avg_scores_colegio.idxmax()
    puntaje_mejor_colegio = avg_scores_colegio.max()

    colegio_peor = avg_scores_colegio.idxmin()
    puntaje_peor_colegio = avg_scores_colegio.min()

    porcentaje_mayor_300 = (filtered_df['punt_global'] > 300).mean() * 100

    # Generar el HTML con las métricas
    metrics = html.Div([
        html.Div(f"Mejor municipio: {municipio_mejor} ({puntaje_mejor_municipio:.2f})"),
        html.Div(f"Peor municipio: {municipio_peor} ({puntaje_peor_municipio:.2f})"),
        html.Div(f"Mejor colegio: {colegio_mejor} ({puntaje_mejor_colegio:.2f})"),
        html.Div(f"Peor colegio: {colegio_peor} ({puntaje_peor_colegio:.2f})"),
        html.Div(f"Porcentaje > 300: {porcentaje_mayor_300:.2f}%"),
    ])

    # Generar opciones para el filtro de municipio
    municipios_options = [{'label': municipio, 'value': municipio} for municipio in filtered_df['cole_mcpio_ubicacion'].unique()]

    return metrics, municipios_options


# Callback para la gráfica de Análisis por Municipio
@app.callback(
    Output('score-graph-municipio', 'figure'),
    [Input('filter-municipio', 'value'),
     Input('filter-periodo', 'value')]
)
def update_municipio_graph(municipio, periodo):
    # Establecer "Manizales" como valor predeterminado
    if not municipio:
        municipio = "Manizales"

    # Filtrar el DataFrame por periodo y municipio
    filtered_df = df if not periodo else df[df['periodo'] == periodo]
    filtered_df = filtered_df[filtered_df['cole_mcpio_ubicacion'] == municipio]

    if filtered_df.empty:
        return px.bar(title="No hay datos disponibles para la selección.")

    # Calcular los promedios por tipo de puntaje
    avg_scores = filtered_df[['punt_global', 'punt_ingles', 'punt_sociales_ciudadanas', 'punt_matematicas', 'punt_c_naturales']].mean()

    # Crear el DataFrame para la gráfica
    avg_scores_df = avg_scores.reset_index()
    avg_scores_df.columns = ['Puntaje', 'Promedio']

    # Crear la gráfica de barras
    return px.bar(
        avg_scores_df,
        x='Puntaje',
        y='Promedio',
        title=f"Puntajes Promedio en {municipio}",
        labels={'Puntaje': 'Tipo de Puntaje', 'Promedio': 'Promedio'},
        text_auto=True
    )

# Callback para la gráfica de Análisis de Desempeño
@app.callback(
    Output('performance-graph', 'figure'),
    [Input('analysis-type', 'value'),
     Input('puntaje-type', 'value'),
     Input('filter-periodo', 'value')]
)
def update_performance_graph(analysis_type, puntaje_type, periodo):
    filtered_df = df if not periodo else df[df['periodo'] == periodo]

    if filtered_df.empty:
        return px.bar(title="No hay datos disponibles para la selección.")

    avg_scores = filtered_df.groupby('cole_mcpio_ubicacion')[puntaje_type].mean().reset_index()
    if analysis_type == 'top_mejores':
        top_5 = avg_scores.sort_values(by=puntaje_type, ascending=False).head(5)
    else:
        top_5 = avg_scores.sort_values(by=puntaje_type, ascending=True).head(5)

    return px.bar(top_5, x='cole_mcpio_ubicacion', y=puntaje_type, 
                  title=f"Top 5 {'Mejor' if analysis_type == 'top_mejores' else 'Peor'} Desempeño")

# Callback para la gráfica de Mapa de Calor
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('heatmap-x-axis', 'value'),
     Input('heatmap-y-axis', 'value'),
     Input('filter-periodo', 'value')]
)
def update_heatmap(x_axis, y_axis, periodo):
    filtered_df = df if not periodo else df[df['periodo'] == periodo]

    if not x_axis or not y_axis or filtered_df.empty:
        return px.imshow([[0]], text_auto=True, title="Seleccione ambas variables y un periodo válido")

    heatmap_data = filtered_df.groupby([x_axis, y_axis])['punt_global'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index=y_axis, columns=x_axis, values='punt_global')
    return px.imshow(heatmap_pivot, text_auto=True, color_continuous_scale='Viridis',
                     title="Mapa de Calor: Puntaje Global Promedio")

# Callback de predicción
@app.callback(
    Output('prediction-output', 'children'),
    Input('calculate-button', 'n_clicks'),
    [Input('model-selection', 'value')] +
    [Input(f'input-{col}', 'value') for col in categorical_columns + numerical_columns]
)
def predict(n_clicks, model_type, *values):
    if n_clicks == 0:
        return "Introduce los datos y presiona el botón para calcular."

    try:
        # Crear un DataFrame con los valores ingresados
        input_data = dict(zip(categorical_columns + numerical_columns, values))
        persona_df = pd.DataFrame([input_data])

        # Seleccionar el scaler y modelo según el tipo
        if model_type == 'classification':
            scaler = scaler_class
            model = model_class
        elif model_type == 'regression':
            scaler = scaler_reg
            model = model_reg
        else:
            return "Tipo de modelo no reconocido."

        # Normalizar las variables numéricas
        persona_df[numerical_columns] = scaler.transform(persona_df[numerical_columns])

        # Realizar la predicción
        if model_type == 'classification':
            probabilidad = model.predict(persona_df.values)[0][0]  # Probabilidad para Keras
            return f"La probabilidad de obtener un puntaje mayor a 300 es: {probabilidad:.4f}"
        elif model_type == 'regression':
            predicted_score = model.predict(persona_df.values)[0][0]  # Predicción continua
            return f"El puntaje predicho del estudiante es: {predicted_score:.2f}"
    except Exception as e:
        return f"Error en la predicción: {str(e)}"

# Inicializar el servidor

if __name__ == '__main__':
    app.run_server(debug=True)