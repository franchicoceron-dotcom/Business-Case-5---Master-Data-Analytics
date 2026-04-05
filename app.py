# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """

Eres un analista de datos experto. Genera código Python (Plotly Express) para analizar este historial de Spotify ({fecha_min} a {fecha_max}).

ESTRUCTURA DEL DATAFRAME `df`:
- `cancion`, `artista`, `album`, `cancion_id`, `ts`.
- `minutos`: Tiempo de escucha en minutos.
- `hora`: (0-23).
- `dia_semana`, `mes`: En inglés (ej. 'Monday', 'January').
- `es_salto`: Booleano (True si el usuario saltó la canción, False si la escuchó completa).
- `platform`: {plataformas}.
- `shuffle`: Booleano.
- `reason_start`: Motivo de comienzo: {reason_start_values}.
- `reason_end`: Motivo de fin: {reason_end_values}.

REGLAS OBLIGATORIAS:
1. Responde solo en JSON con: 
   {{
    "tipo": "grafico" o "fuera_de_alcance",
    "codigo": "import plotly.express as px; fig = px...", 
    "interpretacion": "Breve análisis de 2 frases."
   }}
2. Si piden TIEMPO en periodos largos (como artistas más escuchados), DIVIDE `minutos` entre 60 para mostrar HORAS.
3. El usuario pregunta en español, pero para filtrar por `mes` o `dia_semana`, traduce el valor al INGLÉS (ej: 'Lunes' -> 'Monday').
4. Para "Canciones nuevas", cuenta cuántas `cancion_id` aparecen por primera vez en un periodo.
5. Si la pregunta es ajena a la música, responde en "interpretacion" que está fuera de alcance y deja "codigo" vacío.

"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    df = df[ 
        (df["master_metadata_track_name"].notna()) & 
        (df["master_metadata_album_artist_name"].notna())
    ].copy() # Se filtran registros sin nombre de canción o artista, que suelen ser podcasts u otro tipo de contenido no musical

    df['es_salto'] = df['skipped'].fillna(False).astype(bool) # Se crea una nueva columna booleana 'es_salto' para indicar si la canción fue saltada o no, considerando los NaN como False

    df = df.rename(columns={ # Se renombran columnas para facilitar el trabajo del LLM
        'master_metadata_track_name': 'cancion',
        'master_metadata_album_artist_name': 'artista',
        'master_metadata_album_album_name': 'album',
        'spotify_track_uri': 'cancion_id',
        
    })

    df["ts"] = pd.to_datetime(df["ts"]) # Se convierte la columna de timestamp a formato datetime

    df['hora'] = df['ts'].dt.hour # Se extraen dimensiones temporales para facilitar rankings y patrones
    df['dia_semana'] = df['ts'].dt.day_name() 
    df['mes'] = df['ts'].dt.month_name()      
    df['minutos'] = df['ms_played'] / 60000 # Se convierte el tiempo de reproducción a minutos para facilitar cálculos de duración total


    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    Por un lado, el LLM recibe metadatos, es decir, los nombres de las columnas del dataset 
#    y el contenido dinámico de los placeholders del system prompt, pero nunca el contenido de 
#    los 15.000 registros del archivo JSON.
#    Mientras que, por otro lado, el LLM devuelve un objeto estructurado en formato JSON que contiene 
#    un script de Python y una interpretación textual. Este mismo código se ejecuta localmente en mi 
#    servidor de streamlit gracias a la función exec().
#    Finalmente, el LLM no recibe directamente los datos tanto por una razón de privacidad (los datos 
#    sensibles no salen de mi entorno) como de eficiencia/coste, ya que enviar miles de filas con cada 
#    consulta agotaría el límite de tokens rápidamente. 
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    La información proporcionada al LLM es la estructura exacta del dataframe tras las 
#    transformaciones implementadas. En caso de que la información remitida al LLM sea incompleta o 
#    ambigua, éste alucinará inventándose nombre de columnas, por ejemplo, lo que provocaría un error 
#    de Python al ejecutar el código.
#    Ejemplo de éxito: Gracias a que en el system prompt se incluye la instrucción específica de que 
#    la columna mes contiene valores en inglés (ej. 'January') y que el modelo debe realizar la 
#    traducción antes de filtrar, el LLM genera un código df[df['mes'] == 'January'] que filtra por 
#    los nombres de meses en inglés y no en español, lo que hace que la consulta funcione correctamente.
#    Ejemplo de fallo: Si eliminara del system prompt la descripción de la columna 'es_salto' y su 
#    relación con 'skipped', el LLM no tendría forma de saber que esa columna existe ni qué valores   
#    contiene. Así pues, si el usuario preguntara por "¿Cuántas canciones he saltado cada mes?" el 
#    LLM no podría generar un código.
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    1º El usuario escribe una pregunta en la interfaz de Streamlit.
#    2º La app combina el system prompt con la pregunta del usuario y lanza la petición a la API 
#    de Open AI.
#    3º El modelo gpt-4.1-mini genera el JSON con el código de Python basándose en la estructura del 
#    dataframe definida en el system prompt.
#    4º La app recibe ese JSON del cual extrae el string de la clave “código” y lo ejecuta internamente 
#    contra el dataframe que se cargó con load_data().
#    5º El gráfico resultante se renderiza en pantalla junto con la interpretación textual para el 
#    usuario.
