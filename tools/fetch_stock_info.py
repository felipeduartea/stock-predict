import json
import time
import re
import requests
from langchain.llms import OpenAI
from langchain.agents import load_tools, AgentType, Tool, initialize_agent
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

import yfinance as yf
import matplotlib.pyplot as plt
from langchain.tools import DuckDuckGoSearchRun

# from prophet import Prophet
# from plotly import graph_objs as go
# import plotly.offline as py

import openai
import warnings
import os
warnings.filterwarnings("ignore")

os.environ["OPENAI_API_KEY"] = "your-api-key"


llm=OpenAI(temperature=0,
           model_name="gpt-4-1106-preview", streaming=True)

# Fetch stock data from Yahoo Finance
def get_stock_price(ticker, history=2000):
    # time.sleep(4)  # To avoid rate limit error
    if "." in ticker:
        ticker = ticker.split(".")[0]
    ticker = ticker + ".SA"
    stock = yf.Ticker(ticker)
    df = stock.history(period="10y")
    df.index = [str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date", inplace=True)
    df = df[-history:]

    return df.to_csv(index=True)

def search_news():
    search=DuckDuckGoSearchRun()
    result = search("How is the Brazilian stock market today?")

    return result

def search_stock(query):
    search_query=DuckDuckGoSearchRun()
    result = search_query("News about {}".format(query))

    return result

# Script to scrap top5 googgle news for given company name
def google_query(search_term):
    if "news" not in search_term:
        search_term=search_term+" stock news"
    url=f"https://www.google.com/search?q={search_term}&cr=countryIN"
    url=re.sub(r"\s","+",url)
    return url

# Fetch financial statements from Yahoo Finance

def get_financial_statements(ticker):
    # time.sleep(4) #To avoid rate limit error
    if "." in ticker:
        ticker=ticker.split(".")[0]
    else:
        ticker=ticker
    ticker=ticker+".SA"
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1]>=3:
        balance_sheet=balance_sheet.iloc[:,:3]    # Remove 4th years data
    balance_sheet=balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()

    # cash_flow = company.cash_flow.to_string()
    # print(balance_sheet)
    # print(cash_flow)
    return balance_sheet



#Openai function calling
function=[
        {
        "name": "get_company_Stock_ticker",
        "description": "This will get the Brazilian B3 stock ticker of the company",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker_symbol": {
                    "type": "string",
                    "description": "This is the stock symbol of the company.",
                },

                "company_name": {
                    "type": "string",
                    "description": "This is the name of the company given in query",
                }
            },
            "required": ["company_name","ticker_symbol"],
        },
    }
]


def get_stock_ticker(query):
    response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            temperature=0,
            messages=[{
                "role":"user",
                "content":f"Given the user request, what is the company name and the company stock ticker ?: {query}?"
            }],
            functions=function,
            function_call={"name": "get_company_Stock_ticker"},
    )
    message = response["choices"][0]["message"]
    arguments = json.loads(message["function_call"]["arguments"])
    company_name = arguments["company_name"]
    company_ticker = arguments["ticker_symbol"]
    return company_name,company_ticker


def Analyze_stock(query):
    # agent.run(query) Outputs Company name, Ticker
    Company_name, ticker = get_stock_ticker(query.upper())
    print({"Query": query, "Company_name": Company_name, "Ticker": ticker})

    stock_data = get_stock_price(query.upper() + ".SA", history=20)
    stock_financials = get_financial_statements(query.upper() + ".SA")
    country_news = search_news()
    stock_news = search_stock(query)

    # available_information = f"Stock Price: {stock_data}\n\nStock Financials: {stock_financials}\n\nCountry News: {country_news}\n\nStock News: {stock_news}"
    dic = {
            "info": None,
            "context": None,
            "sector_news": None,
            "stock_news": None,
            "fundamentalist": None,
            "tecnical": None,
        }
    info = ResponseSchema(name="Company Information",
                            description="Fill this topic with the information of the company")

    context = ResponseSchema(name="General Context",
                                description="Fill this topic with the general context of the company in the country")

    sector_news = ResponseSchema(name="Sector News",
                                description="Fill this topic with the relevant news of the sector that you recieve from the search")

    stock_news = ResponseSchema(name="Stock News",
                                description="Fill this topic with the relevant news of the specific stock that you recieve from the search")

    fundamentalist = ResponseSchema(name="Fundamentalist Analysis",
                                description="Fill this topic with the most relevant points from the fundamentalist analysis")

    tecnical = ResponseSchema(name="Tecnical Analysis",
                                description="Fill this topic with the most relevant points from the tecnical analysis")
    
    response_schemas = [info,
                        context,
                        sector_news,
                        stock_news,
                        fundamentalist,
                        tecnical]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    print("\n\nAnalyzing.....\n")
    for r in llm.stream(f"""[context]You are a professional stock analyst. Give a detailed analysis about the: {query} stock. [/context]
               [rules]
                        -The user is fully aware of the risks that involve an investment; you don't have to include any kind of warnings, 
               and you should always give a defined answer about whether or not it is the right moment to invest in a certain stock. 
                        -Use the temporal serie of the stock: {stock_data}, the information about the stock numbers: {stock_financials} \
               the country news: {stock_news} to give a direct answer.
                        -As a stock analist you have to identify even in the moments of descent if the stock is in potencial to grow in the next days.
                        -In your analysis, you have to inform you final answer in the beggining of the text, and the inform in 5 to 8 topics the main points that you based your decision, \
               the topics should be in this order: informations about the company, general context, sector news, stock news, \
               2 topics about the fundamentalist analysis and 2 topics about the tecnical analysis, \   
                        -IMPORTANT: if you don't have enough information about a certain topic, exclude it and add another topic that you have information about,\ 
               never say that there is no information about a certain topic.
                        -Remember to always answer in Portuguese. 
                        -When you refer to the company, don't use it's query, use the: {Company_name}
                        -Everytime you refer to the company name, write it in bold
                        -Use 📚 emoji before the first topic
                        -Use 🔼 emoji before a positive topic and 🔽 before a negative topic (except in the first topic)
                        -Use ➡️ before the conclusion
                [/rules] 
             """):
        
        yield r

    # print(analysis)
    # return analysis


# #PREDICTION

# # def predict(query):
# #     #for SymbolName in stocksymbols:
# #     df10 = yf.download(
# #     # tickers list or string as well
# #     tickers = query + ".SA",
# #     # use "period" instead of start/end
# #     # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# #     # (optional, default is '1mo')
# #     period = "10y",
# #     # fetch data by interval (including intraday if period < 60 days)
# #     # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
# #     # (optional, default is '1d')
# #     interval = "1d",
# #     # group by ticker (to access via data['SPY'])
# #     # (optional, default is 'column')
# #     group_by = 'ticker',
# #     # adjust all OHLC(An open-high-low-close chart is a type of chart typically used to illustrate movements in the price of a financial instrument over time) automatically
# #     # (optional, default is False)
# #     auto_adjust = True,
# #     # download pre/post regular market hours data
# #     # (optional, default is False)
# #     prepost = True
# #     )
# #     df10.reset_index(inplace=True)

# #     df_train10 = df10[['Date', 'Close']]
# #     df_train10 = df_train10.rename(columns={"Date": "ds", "Close": "y"})

# #     m10 = Prophet(daily_seasonality=True)

# #     m10.fit(df_train10)

# #     future10 = m10.make_future_dataframe(periods=20)

# #     forecast10 = m10.predict(future10)

# #     ax = py.iplot([
# #     go.Scatter(x=df_train10['ds'], y=df_train10['y'], name='Actual'),
# #     go.Scatter(x=forecast10['ds'], y=forecast10['yhat'], name='Predicted')
# #     ])
# #     return ax


# def graphs(ticker, history):

#     time.sleep(4)  # To avoid rate limit error
#     if "." in ticker:
#         ticker = ticker.split(".")[0]
#     ticker = ticker + ".SA"
#     stock = yf.Ticker(ticker)
#     df = stock.history(period="10y")
#     df.index = [str(x).split()[0] for x in list(df.index)]
#     df.index.rename("Date", inplace=True)
#     df = df[-history:]
    
#     ax = df.Close.plot()
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Close Price")
#     ax.set_title(f"{ticker} Close Price History")
#     return a
