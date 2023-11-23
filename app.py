import streamlit as st
import tools
from tools.fetch_stock_info import Analyze_stock
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
import plotly.offline as py
from prophet import Prophet
import pandas as pd
import plotly.graph_objs as go
import time
st.title("INVEST.IA")
st.write("DISCLAIMER: Essa LLM não é uma consultora financeira. Ela apenas analisa dados públicos e fornece informações para auxiliar na tomada de decisão. O usuário é o único responsável por suas decisões financeiras.")

# Dropdown menu for stock selection
selected_stock = st.selectbox('Selecione uma ação:', ['BBAS3', 'AZUL4', 'PETR4', 'VALE3','BHIA3','OIBR3','ITSA4','ABEV3','RAPT3','VITT3', "XPML11", "BRCO11"])

Enter = st.button("Enter")
clear = st.button("Clear")

if clear:
    st.markdown(' ')

if Enter:
    # Map selected stock to its corresponding ticker symbol
    stock_mapping = {'BBAS3': 'BBAS3.SA', 'AZUL4': 'AZUL4.SA', 'PETR4': 'PETR4.SA', 'VALE3': 'VALE3.SA', 'BHIA3': 'BHIA3.SA', 'OIBR3': 'OIBR3.SA', 'ITSA4': 'ITSA4.SA', 'ABEV3': 'ABEV3.SA', 'RAPT3': 'RAPT3.SA', 'VITT3': 'VITT3.SA', "XPML11": "XPML11.SA", "BRCO11": "BRCO11.SA"}
    query = stock_mapping.get(selected_stock, '')
    
    if not query:
        st.warning('Por favor, selecione uma ação válida.')
    else:
        history = 2000
        time.sleep(4)  # To avoid rate limit error
        if "." in query:
            query = query.split(".")[0]
        query = query + ".SA"
        stock = yf.Ticker(query)
        df = stock.history(period="10y")
        df.index = [str(x).split()[0] for x in list(df.index)]
        df.index.rename("Date", inplace=True)
        df['SMA50'] = df['Close'].rolling(50).mean()
        df = pd.concat([df, df['SMA50']], axis=1)

        df = df[-history:]

        # Plot candlestick chart
        fig_candlestick = go.Figure(data=[go.Candlestick(x=df.index,
                                                        open=df['Open'],
                                                        high=df['High'],
                                                        low=df['Low'],
                                                        close=df['Close'])])

        fig_candlestick.update_layout(title=f'Série Temporal da {query}',
                                    xaxis_title='Date',
                                    yaxis_title='Stock Price',
                                    xaxis_rangeslider_visible=False)

        # Show the candlestick chart in Streamlit
        st.plotly_chart(fig_candlestick)
        history = 5000
        time.sleep(4)  # To avoid rate limit error
        if "." in query:
            query = query.split(".")[0]
        query = query + ".SA"
        stock = yf.Ticker(query)
        df = stock.history(period="10y")
        df.index = [str(x).split()[0] for x in list(df.index)]
        df.index.rename("Date", inplace=True)
        df = df[-history:]

        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df_train10 = df[['Date', 'Close']]
        df_train10 = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        m10 = Prophet(daily_seasonality=True)

        m10.fit(df_train10)

        future10 = m10.make_future_dataframe(periods=50)

        forecast10 = m10.predict(future10)

        # # Creating a figure and axis
        # fig1, ax1 = plt.subplots()

        df_train10 =  df_train10.loc['2023-06-01':]
        forecast10 = forecast10.loc['2023-06-01':]

        # # Plotting historical data
        # ax1.plot(df_train10['ds'], df_train10['y'], label='Historical Data')

        # # Plotting forecasted values
        # ax1.plot(forecast10['ds'], forecast10['yhat'], label='Forecasted Data')

        # Creating a figure using Plotly for Prophet prediction graph
        fig1 = go.Figure()

        # Plotting historical data
        fig1.add_trace(go.Scatter(x=df_train10['ds'], y=df_train10['y'], mode='lines', name='Historical Data'))

        # Plotting forecasted values with green color
        fig1.add_trace(go.Scatter(x=forecast10['ds'], y=forecast10['yhat'], mode='lines', name='Forecasted Data', line=dict(color='green')))

        # Updating layout
        fig1.update_layout(xaxis_title='Dia',
                        yaxis_title='Valor de Fechamento Futuro',
                        title=f'Análise Preditiva dos Valores de Fechamento da {query}',
                        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'))

        # Display the Prophet prediction graph in Streamlit
        st.plotly_chart(fig1)

        full_response = ""
        ph = st.empty()
        for r in Analyze_stock(query):
            full_response += r
            ph.markdown(full_response + "|")



