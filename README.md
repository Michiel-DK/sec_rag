# SEC 10-K Company Similarity Explorer

AI-powered company analysis using SEC 10-K filings and vector similarity search.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file with your Google API key:
```bash
GOOGLE_API_KEY=your-api-key-here
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```

**Default login credentials:**
- Username: `demo`
- Password: `changeme123`

### 4. Configure Authentication (Important!)

**For local development:**
Edit `.streamlit/secrets.toml`:
```toml
[auth]
username = "your-username"
password = "your-password"
```

**For Streamlit Cloud deployment:**
1. Go to your app settings on Streamlit Cloud
2. Click "Secrets" in the sidebar
3. Add:
```toml
[auth]
username = "your-username"
password = "your-password"
```

## Features

- 🔍 Find similar companies across 37 S&P 500 companies
- 📊 Compare companies by business model, risk profile, financials, etc.
- 🔐 Password-protected access
- 💬 Natural language queries
- 🎯 Extract directors, products, risks from SEC filings

## Available Companies

Tech: AAPL, MSFT, GOOG, META, NVDA, TSLA, AMD, AVGO, CSCO, ORCL, CRM, IBM, PLTR  
Finance: MA, V, AXP, MS, WFC, JPM, BAC  
Healthcare: ABBV, ABT, JNJ, LLY, UNH  
Consumer: COST, WMT, HD, KO, PG, PM  
Energy: CVX, XOM  
Other: AMZN, GE, NFLX, TMUS, CCZ

## Example Queries

- "Find companies similar to TSLA"
- "Why are AAPL and MSFT similar?"
- "Who are the directors at GOOG?"
- "What are the main risks for NVDA?"
- "Compare AAPL and MSFT in business model"

## Deployment to Streamlit Cloud

1. Push code to GitHub (including `chroma_db/` folder)
2. Connect repository on [streamlit.io](https://streamlit.io)
3. Set secrets in app settings
4. Deploy!

## Project Structure

```
sec_rag/
├── app.py                    # Streamlit application
├── sec_rag/
│   ├── agent/               # ReAct agent & tools
│   ├── similarity/          # Similarity engine
│   ├── llm/                 # LLM explainer
│   └── chroma/              # Vector DB utilities
├── chroma_db/               # Vector database (219MB)
└── requirements.txt
```

## Security Note

⚠️ **Never commit `.streamlit/secrets.toml` to git!** It's already in `.gitignore`.

For production, use stronger passwords and consider implementing proper authentication (OAuth, etc.).
