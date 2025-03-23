import os
import json
import requests
import yfinance as yf
from datetime import datetime, timedelta

from langchain_huggingface import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API Keys
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Step 1: Fetch Stock Data (52-Week High Validation)
def get_yfinance_data(symbol):
    stock = yf.Ticker(symbol)
    
    # Get historical data for the last 12 months
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    history = stock.history(start=start_date, end=end_date)
    
    if history.empty:
        return {"error": "No data found for this stock."}

    # Calculate the 52-week high
    high_52_week = history["High"].max()
    
    # Get last month's data
    last_month_start = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    last_month_data = history[last_month_start:]
    
    if last_month_data.empty:
        return {"error": "No data found for the last month."}

    last_month_high = last_month_data["High"].max()
    last_month_low = last_month_data["Low"].min()
    last_month_close = last_month_data["Close"].iloc[-1]
    last_month_open = last_month_data["Open"].iloc[0]
    last_month_volume = last_month_data["Volume"].sum()

    # Percentage change over last month
    percentage_change = ((last_month_close - last_month_open) / last_month_open) * 100

    # Short summary
    short_summary = f"""
    📊 **{symbol} - Last Month Summary** 📊
    - **High:** {last_month_high:.2f}
    - **Low:** {last_month_low:.2f}
    - **Close Price:** {last_month_close:.2f}
    - **% Change:** {percentage_change:.2f}%
    - **Total Trading Volume:** {last_month_volume:,}
    """

    return {
        "symbol": symbol,
        "52_Week_High": high_52_week,
        "Last_Month_High": last_month_high,
        "Last_Month_Low": last_month_low,
        "Last_Close": last_month_close,
        "Percentage_Change": percentage_change,
        "Trading_Volume": last_month_volume,
        "Short_Summary": short_summary,
    }


# Step 2: Alpha Vantage API Data Fetching
def get_alpha_vantage_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data

# Step 3: Initialize LLM (Zephyr-7B via Hugging Face API)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature=0.5,
    max_new_tokens= 512,
    repetition_penalty = 1.1,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Step 4: Summarization Function
import numpy as np

def json_safe_data(data):
    """Convert NumPy int64/float64 types to standard Python types."""
    if isinstance(data, dict):
        return {key: json_safe_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [json_safe_data(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)  # Convert NumPy int to Python int
    elif isinstance(data, np.floating):
        return float(data)  # Convert NumPy float to Python float
    else:
        return data  # Keep other types unchanged

def summarize_data(data):
    """Uses LLM to generate a structured summary of stock data."""
    
    safe_data = json_safe_data(data)  #  Ensure all values are JSON serializable
    
    summary_prompt = f"""
    📊 **Stock Summary Request**
    
    **Stock Data:**
    {json.dumps(safe_data, indent=2)}
    
    **Instructions for LLM:**
    - Summarize this stock data in **3 bullet points**.
    - Mention **market sentiment, investment risks, and final recommendation**.
    """
    
    return llm.invoke(summary_prompt)



# Step 5: Store and Retrieve Summaries in FAISS (RAG)
embedding_model = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_key=HUGGINGFACEHUB_API_TOKEN
)
vector_store = FAISS.from_texts(
    texts=["Placeholder text for initialization"],  
    embedding=embedding_model
)

def store_in_vector_db(symbol, summary):
    """Stores the stock summary in FAISS for retrieval."""
    vector_store.add_texts(texts=[summary], metadatas=[{"symbol": symbol}])

def retrieve_from_vector_db(query):
    """Retrieves relevant stock data from FAISS based on user query."""
    docs = vector_store.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No relevant stock data found."

# Step 6: Define LangChain Tools
alpha_vantage_tool = Tool(
    name="AlphaVantage",
    func=get_alpha_vantage_data,
    description="Fetch stock market data from Alpha Vantage."
)
yfinance_tool = Tool(
    name="YahooFinance",
    func=get_yfinance_data,
    description="Fetch additional stock insights from Yahoo Finance."
)

# Step 7: Memory for Conversation History
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Step 8: Define the Agent
agent_executor = initialize_agent(
    tools=[alpha_vantage_tool, yfinance_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# Step 9: Run Agent & Store Data
user_query = input("Enter a stock symbol: ")
stock_data = get_yfinance_data(user_query)

if "error" not in stock_data:
    print("\n📈 Short Summary of Last Month:\n")
    print(stock_data["Short_Summary"])  # Display short summary

    # Generate and display LLM summary regardless of 52-week high status
    summary = summarize_data(stock_data)
    store_in_vector_db(user_query, summary)

    print("\n📊 Summarized Stock Data (LLM Generated):\n")
    print(summary)  



# Step 10: Q&A with RAG-Based Context
def answer_user_query(user_query):
    """Retrieves relevant stock data, formats it, and uses LLM for Q&A."""
    
    retrieved_text = retrieve_from_vector_db(user_query)
    
    # New prompt structure for better answers
    prompt = f"""
    You are a financial analyst. The user has a question about a stock.
    
    **Stock Data Retrieved:**
    {retrieved_text}
    
    **User's Question:**
    {user_query}
    
    🔹 **Your Task:**
    - Analyze the stock data.
    - Provide financial insights, risks, and trends.
    - Give a structured response.
    - DO NOT just repeat the numbers—explain their meaning.
    """
    
    return llm.invoke(prompt)

# Step 11: Interactive Q&A Session
while True:
    user_question = input("Ask a question about the stock data (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        break
    response = answer_user_query(user_question)
    print(response)

