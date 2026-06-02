from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from portfolio import main as analyze_portfolio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PortfolioRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str

    @validator("tickers")
    def validate_tickers(cls, v):
        if len(v) < 2:
            raise ValueError("Please provide at least 2 tickers")
        if len(v) > 10:
            raise ValueError("Maximum 10 tickers allowed")
        return [t.upper().strip() for t in v]

    @validator("end_date")
    def validate_dates(cls, end_date, values):
        try:
            start = datetime.strptime(values["start_date"], "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")

        if end <= start:
            raise ValueError("End date must be after start date")
        if (end - start).days < 30:
            raise ValueError("Date range must be at least 30 days")
        if end > datetime.today():
            raise ValueError("End date cannot be in the future")

        return end_date

@app.post("/analyze")
def analyze(request: PortfolioRequest):
    try:
        # ── validate tickers exist ────────────────────────────
        test = yf.download(
            request.tickers,
            start=request.start_date,
            end=request.end_date,
            progress=False
        )["Close"]

        if test.empty:
            raise HTTPException(
                status_code=400,
                detail="No data found — check your ticker symbols"
            )

        # ── check for all NaN ─────────────────────────────────
        if test.isnull().all().all():
            raise HTTPException(
                status_code=400,
                detail="Data returned all NaN — try different tickers or date range"
            )

        # ── check invalid tickers ─────────────────────────────
        invalid = [col for col in test.columns if test[col].isnull().all()]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or unavailable tickers: {invalid}"
            )

        # ── run analysis ──────────────────────────────────────
        result = analyze_portfolio(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/")
def root():
    return {"status": "API is running"}