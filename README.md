# 🧠 Deep Research Agent - 52-Week High Stock Analysis  

## 📌 Overview  
This AI-powered research agent fetches financial data for stocks, analyzes **52-week highs**, and summarizes insights using **Zephyr-7B** LLM.

## 🚀 Features  
✅ Fetch stock data from **Alpha Vantage** & **Yahoo Finance**  
✅ Identify **52-week high** stocks in the Indian market  
✅ Generate **structured stock insights**  
✅ Answer **user queries** about stocks  

## 📦 Installation  
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DeepResearchAgent.git
   cd DeepResearchAgent

2. Install dependencies:

pip install -r requirements.txt

3. Set up environment variables:
cp .env.example .env

4. Open .env and add:

ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token

5. Run the agent:

python agent.py

📝 Example Usage

Enter a stock symbol: RELIANCE.NS

📈 Short Summary of Last Month:

📊 **RELIANCE.NS - Last Month Summary** 📊
- **High:** 2,890.50
- **Low:** 2,760.10
- **Close Price:** 2,865.20
- **% Change:** +3.21%
- **Total Trading Volume:** 45,000,000

📊 Summarized Stock Data (LLM Generated):
- **Market Sentiment:** Positive 📈
- **Investment Risks:** Moderate Volatility
- **Final Recommendation:** HOLD


📄 Contributors
Shashank Vashisht - AI Engineer 
