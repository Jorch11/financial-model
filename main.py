import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"                               #Día inicial para el modelo
TODAY = date.today().strftime("%Y-%m-%d")          #Día actual



st.title("Modelo para pronosticar Acciones")

accion = ("AAPL", "GOOG", "MSFT", "GME","CELSIA.CL","^GSPC","TSLA","META","EC")
selected_stock = st.selectbox("Selecciona una compañía para pronosticar", accion)             #Función de st para seleccionar de Acción

n_years = st.slider("Años de predicción: ", 1, 4)
period = n_years * 365

@st.cache_data                         # usa la cache para no tener que volver a cargar esto al cambiar de acción
def cargar_datos(ticker):
    """Función para cargar datos
    var data = guarda los datos que se descargan desde yfinance, ticker es lo que entre, star el dia que inicia y today la fecha actual"""
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)  #Resetea el indice y pone los dias encambio
    return data

data_load_state = st.text("Cargando datos...")
data = cargar_datos(selected_stock)
data_load_state.text("Cargando datos... Hecho!")

st.subheader('Histórico')
st.write(data.tail())

def plot_historico():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name= 'stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_historico()
def forecast():
    #Forecasting
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns = {"Date":"ds","Close":"y"}) #Renombrar las columnas porque asi lo requiere el prophet

    m = Prophet()                         #Crear instancia
    m.fit(df_train)                       #Entrenar el modelo
    future = m.make_future_dataframe(periods=period)    #Se hace el forecast
    forecast = m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.write('forecast data')
    fig1 = plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    st.write('forecast components')
    fig2 = m.plot_components((forecast))
    st.write(fig2)

st.title("Pronosticar")
result = st.button("Haz click acá!")
if result:
    forecast()