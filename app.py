
import streamlit as st
import pandas as pd
import joblib

# Cargar el Modelo 
# ---------------------------------
# @st.cache_resource para que el modelo se cargue solo una vez
# y no en cada re-ejecución de la app.
@st.cache_resource
def load_model():
    """Carga el modelo K-Means desde el archivo .pkl"""
    try:
        model = joblib.load('kmeans_clientes.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: Archivo 'kmeans_clientes.pkl' no encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

# Cargar el modelo al inicio
kmeans_model = load_model()


# Definir los Segmentos 
# ---------------------------------

segment_map = {
    0: 'Estándar Promedio (Ingreso Medio, Gasto Medio)',
    1: 'Objetivo VIP (Ingreso Alto, Gasto Alto)',
    2: 'Derrochadores (Ingreso Bajo, Gasto Alto)',
    3: 'Ahorradores Cuidadosos (Ingreso Alto, Gasto Bajo)',
    4: 'Promedio Cautelosos (Ingreso Bajo, Gasto Bajo)'
}


# Interfaz de la App
# ---------------------------------
st.title('🤖 Segmentador de Clientes (K-Means)')
st.write("""
Esta aplicación utiliza un modelo de Machine Learning (K-Means)
para predecir el segmento de mercado de un cliente del centro comercial
basado en su ingreso y puntaje de gasto.
""")
st.write("---")

# --- Barra Lateral (Inputs) ---
st.sidebar.header('📊 Ingresar Datos del Cliente')

ingreso = st.sidebar.slider(
    'Ingreso Anual (en miles de US$)',
    min_value=15,   # Valor mínimo del dataset
    max_value=140,  # Valor máximo del dataset
    value=50,       # Valor por defecto
    step=1
)

puntaje = st.sidebar.slider(
    'Puntaje de Gasto (1-100)',
    min_value=1,
    max_value=100,
    value=50,
    step=1
)

# --- Predicción y Resultados ---
if st.sidebar.button('Segmentar Cliente'):
    # 1. Preparar los datos de entrada para el modelo
    # El modelo espera un DataFrame con los mismos nombres de columnas
    # usado para entrenar ('Ingreso_Anual', 'Puntaje_Gasto')
    input_data = pd.DataFrame({
        'Ingreso_Anual': [ingreso],
        'Puntaje_Gasto': [puntaje]
    })

    # 2. Realizar la predicción
    try:
        prediction = kmeans_model.predict(input_data)
        cluster_id = prediction[0]  

        # 3. Obtener la descripción del segmento
        segment_name = segment_map.get(cluster_id, "Cluster Desconocido")

        # 4. Mostrar el resultado
        st.success(f"### ¡Predicción Exitosa!")
        st.metric(label="Cluster Asignado:", value=f"Cluster {cluster_id}")
        st.subheader(f"Tipo de Cliente: **{segment_name}**")
        
        st.write("---")
        st.write("Datos de entrada utilizados:")
        st.dataframe(input_data, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")

else:
    st.info('Ajusta los sliders en la barra lateral y presiona "Segmentar Cliente" para ver el resultado.')


