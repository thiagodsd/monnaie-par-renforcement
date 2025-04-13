import pandas as pd
from pathlib import Path
import numpy as np

def load_and_prepare_data(window_size: int = 10) -> pd.DataFrame:
    """Carrega e integra todos os dados para criar a ABT"""
    data_dir = Path("../data/02_intermediate")
    
    # Carregar todos os datasets
    btc = pd.read_parquet(data_dir / "historical_data_btc.parquet")
    fed = pd.read_parquet(data_dir / "us_fed_rate.parquet")
    sp500 = pd.read_parquet(data_dir / "SP500.parquet")
    fng = pd.read_parquet(data_dir / "greed_and_fear.parquet")
    djia = pd.read_parquet(data_dir / "DJIA.parquet")
    djca = pd.read_parquet(data_dir / "DJCA.parquet")
    
    # Converter timestamps e datas
    btc['date'] = pd.to_datetime(btc['formatted_date'])
    fed['date'] = pd.to_datetime(fed['Release Date'])
    sp500['date'] = pd.to_datetime(sp500['observation_date'])
    fng['date'] = pd.to_datetime(fng['date'])
    djia['date'] = pd.to_datetime(djia['observation_date'])
    djca['date'] = pd.to_datetime(djca['observation_date'])
    
    # Combinar todos os dados na mesma frequência diária
    df = btc[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.merge(fed[['date', 'Actual']], on='date', how='left')
    df = df.merge(sp500[['date', 'SP500']], on='date', how='left')
    df = df.merge(fng[['date', 'fng_value']], on='date', how='left')
    df = df.merge(djia[['date', 'DJIA']], on='date', how='left')
    df = df.merge(djca[['date', 'DJCA']], on='date', how='left')
    
    # Preencher valores faltantes
    df['Actual'] = df['Actual'].str.replace('%', '').astype(float).ffill()
    df[['SP500', 'DJIA', 'DJCA', 'fng_value']] = df[
        ['SP500', 'DJIA', 'DJCA', 'fng_value']].ffill()
    
    # Engenharia de features
    df['price_change_pct'] = df['close'].pct_change() * 100
    df['volatility'] = df['high'] - df['low']
    
    # Indicadores técnicos
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_21'] = df['close'].rolling(window=21).mean()
    df['rsi'] = calculate_rsi(df['close'], window=14)
    
    # Features de mercado
    df['sp500_change'] = df['SP500'].pct_change() * 100
    df['djia_change'] = df['DJIA'].pct_change() * 100
    
    # Remover linhas com valores faltantes
    df = df.dropna().reset_index(drop=True)
    
    # Features temporais
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Janela deslizante para features sequenciais
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'price_change_pct', 'volatility', 'Actual',
        'SP500', 'DJIA', 'DJCA', 'fng_value',
        'sma_7', 'sma_21', 'rsi', 'sp500_change',
        'djia_change', 'day_of_week', 'month'
    ]
    
    # Criar janelas temporais
    for feature in features:
        for window in [1, 3, 5]:
            df[f'{feature}_delta_{window}'] = df[feature].diff(window)
            
    return df.dropna()

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calcula o Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
