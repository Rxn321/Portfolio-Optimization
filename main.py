#fastApi file

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf

from portfolio import main as analyze_portfolio

app = FastAPI()

# ── CORS (allows your React site to call this API) ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # URl
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request model ─────────────────────────────────────────────
class PortfolioRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str

# ── Main endpoint ─────────────────────────────────────────────
@app.post("/analyze")
def analyze(request: PortfolioRequest):
    result = analyze_portfolio(
        tickers=request.tickers,
        start_date=request.start_date,
        end_date=request.end_date
    )
    return result

# API check
@app.get("/")
def root():
    return {"status": "API is running"}