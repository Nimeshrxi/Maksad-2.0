import os
import pandas as pd
import numpy as np
import talib
import logging
import time
import asyncio
import json
import sys
import signal
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical   
import joblib
import MetaTrader5 as mt5
import sqlite3
from uuid import uuid4
from imblearn.over_sampling import SMOTE

# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logger = logging.getLogger('xauusd_trader')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "trade_id": "%(trade_id)s", "message": "%(message)s"}')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler('xauusd_trade.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
debug_handler = logging.FileHandler('xauusd_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)
logger.addHandler(debug_handler)

# Configuration
CONFIG = {
    "SYMBOLS": ["XAUUSD"],
    "TIMEFRAMES": {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
    },
    "INITIALEQUITY": 10000.0,
    "RISK_PER_TRADE_PERCENT": 1.0,
    "MAX_DRAWDOWN_PERCENT": 5.0,
    "MLTRAIN_INTERVAL": timedelta(seconds=15),
    "TARGET_PROFIT_PERCENT": 0.3,
    "MIN_PROFIT_PERCENT": 0.05,
    "MT5_LOGIN": int(os.getenv("MT5_LOGIN", 5038204514)),
    "MT5_PASSWORD": os.getenv("MT5_PASSWORD", "P_MzRpM6"),
    "MT5_SERVER": os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
    "MAX_OPEN_TRADES_PER_SYMBOL": 1,
    "SIGNAL_THRESHOLD": 0.2,
    "MIN_STOP_DISTANCE_FACTOR": 2.0,
    "PRICE_CHANGE_THRESHOLD": 0.0003,
    "MIN_LOT_SIZE": 0.01,
    "MAX_LOT_SIZE": 0.01,
    "MIN_BARS": 700,
    "MAX_RETRIES": 3,
    "RETRY_BACKOFF_FACTOR": 2,
    "MAX_ERRORS": 5,
    "DEFAULT_ATR": 0.0002,
    "DIAGNOSTIC_MODE": False,
    "RSI_BUY_THRESHOLD": 45,
    "RSI_SELL_THRESHOLD": 55,
    "FORCE_RULE_BASED": False,
    "SMA_PERIOD": 50,
    "DIVERGENCE_LOOKBACK": 40,
    "FORCE_TEST_TRADE": False,
    "TEST_TRADE_REASON": "Debugging purposes",
    "MIN_BARS_FOR_TRAINING": 300,
    "ML_MODEL_SAVE_INTERVAL": timedelta(seconds=5),
    "ML_MODEL_LOAD_INTERVAL": timedelta(seconds=10),
    "ML_MODEL_RETRAIN_INTERVAL": timedelta(seconds=30),
    "ML_MODEL_RETRAIN_THRESHOLD": 0.05,
    "ML_MODEL_PERFORMANCE_METRIC": "accuracy",
    "ML_MODEL_PERFORMANCE_THRESHOLD": 0.6,
    "ML_MODEL_FEATURES": ['rsi', 'macd', 'macd_signal', 'atr', 'price_change_pct', 'volatility'],
    "ML_MODEL_TARGET": 'signal',
    "ML_MODEL_CLASS_WEIGHTS": {0: 1, 1: 2, 2: 3},
    "ML_MODEL_UNDER_SAMPLING": True,
    "ML_MODEL_OVER_SAMPLING": True,
    "ML_MODEL_SMOTE_K_NEIGHBORS": 5,
    "ML_MODEL_SMOTE_RANDOM_STATE": 42,
    "ML_MODEL_XGB_PARAMS": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": 'multi:softprob',
        "num_class": 3,
        "random_state": 42
    },
    "ML_MODEL_XGB_EVAL_METRIC": 'mlogloss',
    "ML_MODEL_XGB_EARLY_STOPPING_ROUNDS": 10,
    "ML_MODEL_XGB_VERBOSITY": 1,
    "ML_MODEL_XGB_GPU": False,
    "ML_MODEL_XGB_GPU_ID": 0,
    "ML_MODEL_XGB_TREE_METHOD": 'auto',
    "ML_MODEL_XGB_BOOSTING_TYPE": 'gbtree',
    "ML_MODEL_XGB_REGULARIZATION": {
        "lambda": 1.0,
        "alpha": 0.0
    },
    "ML_MODEL_XGB_SCALE_POS_WEIGHT": 1.0,
    "ML_MODEL_XGB_RANDOM_SEED": 42,
    "ML_MODEL_XGB_NUM_BOOST_ROUNDS": 1000,
    "ML_MODEL_XGB_TRAIN_TEST_SPLIT": 0.8,
    "ML_MODEL_XGB_GRID_SEARCH": {
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        },
        "cv": 3,
        "scoring": 'accuracy',
        "verbose": 1,
        "n_jobs": -1
    }
}

# Global variables
trade_stats = {symbol: {"wins": 0, "losses": 0, "total_trades": 0, "net_profit": 0.0} for symbol in CONFIG["SYMBOLS"]}
ml_models = {symbol: {'primary': None} for symbol in CONFIG["SYMBOLS"]}
ml_trained = {symbol: {'primary': True} for symbol in CONFIG["SYMBOLS"]}
last_ml_train = {symbol: {'primary': datetime.now() - CONFIG["MLTRAIN_INTERVAL"]} for symbol in CONFIG["SYMBOLS"]}
trade_history = []
monitored_positions = []
mt5_initialized = False
error_count = 0
trade_id_context = {'trade_id': ''}

def init_db():
    try:
        with sqlite3.connect('trade_history.db') as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    action TEXT,
                    price_open REAL,
                    volume REAL,
                    sl REAL,
                    tp REAL,
                    ticket TEXT,
                    signal_strength REAL,
                    strategy_type TEXT,
                    price_close REAL,
                    profit REAL,
                    reason TEXT
                )
            ''')
            conn.commit()
        logger.info("SQLite database initialized.", extra=trade_id_context)
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}", extra=trade_id_context)

def save_state():
    try:
        with sqlite3.connect('trade_history.db') as conn:
            for trade in trade_history:
                c = conn.cursor()
                c.execute('''
                    INSERT OR REPLACE INTO trades (
                        trade_id, timestamp, symbol, action, price_open, volume, sl, tp, ticket,
                        signal_strength, strategy_type, price_close, profit, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.get('trade_id', str(uuid4())),
                    trade.get('timestamp', ''),
                    trade.get('symbol', ''),
                    trade.get('action', ''),
                    trade.get('price_open', 0.0),
                    trade.get('volume', 0.0),
                    trade.get('sl', 0.0),
                    trade.get('tp', 0.0),
                    trade.get('ticket', ''),
                    trade.get('signal_strength', 0.0),
                    trade.get('strategy_type', ''),
                    trade.get('price_close', 0.0),
                    trade.get('profit', 0.0),
                    trade.get('reason', '')
                ))
            conn.commit()
        logger.info("State saved.", extra=trade_id_context)
    except Exception as e:
        logger.error(f"Save state error: {str(e)}", extra=trade_id_context)

def save_ml_model(symbol):
    try:
        if ml_models[symbol]['primary'] is None:
            return
        joblib.dump(ml_models[symbol]['primary']['model'], f'{symbol}_model.joblib')
        joblib.dump(ml_models[symbol]['primary']['scaler'], f'{symbol}_scaler.joblib')
        logger.info(f"ML model for {symbol} saved.", extra=trade_id_context)
    except Exception as e:
        logger.error(f"Save ML model error for {symbol}: {str(e)}", extra=trade_id_context)

def load_ml_model(symbol):
    global ml_models, ml_trained
    try:
        if not os.path.exists(f'{symbol}_model.joblib') or not os.path.exists(f'{symbol}_scaler.joblib'):
            logger.warning(f"ML model files missing for {symbol}. Retraining required.", extra=trade_id_context)
            return False
        model = joblib.load(f'{symbol}_model.joblib')
        scaler = joblib.load(f'{symbol}_scaler.joblib')
        ml_models[symbol]['primary'] = {'model': model, 'scaler': scaler}
        ml_trained[symbol]['primary'] = True
        logger.info(f"ML model for {symbol} loaded.", extra=trade_id_context)
        return True
    except Exception as e:
        logger.error(f"Load ML model error for {symbol}: {str(e)}", extra=trade_id_context)
        return False

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}. Shutting down...", extra=trade_id_context)
    graceful_shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def graceful_shutdown():
    try:
        save_state()
        if mt5_initialized:
            mt5.shutdown()
        logger.info("Shutdown completed.", extra=trade_id_context)
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", extra=trade_id_context)

def initialize_mt5():
    global mt5_initialized
    try:
        logger.info(f"Initializing MT5 with server: {CONFIG['MT5_SERVER']}", extra=trade_id_context)
        if mt5.initialize(
            login=CONFIG["MT5_LOGIN"],
            password=CONFIG["MT5_PASSWORD"],
            server=CONFIG["MT5_SERVER"],
            timeout=60000
        ):
            for symbol in CONFIG["SYMBOLS"]:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol}: {mt5.last_error()}", extra=trade_id_context)
                    mt5.shutdown()
                    return False
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None or not symbol_info.visible:
                    logger.error(f"Symbol {symbol} not found or not visible: {mt5.last_error()}", extra=trade_id_context)
                    mt5.shutdown()
                    return False
                logger.info(f"Symbol {symbol} details: visible={symbol_info.visible}, session_deals={symbol_info.session_deals}, point={symbol_info.point}, digits={symbol_info.digits}, trade_stops_level={symbol_info.trade_stops_level}, volume_min={symbol_info.volume_min}", extra=trade_id_context)
            mt5_initialized = True
            logger.info("MT5 initialized successfully.", extra=trade_id_context)
            return True
        error = mt5.last_error()
        logger.error(f"MT5 initialization failed: Code={error[0]}, Message={error[1]}", extra=trade_id_context)
        return False
    except Exception as e:
        logger.error(f"MT5 initialization error: {str(e)}", extra=trade_id_context)
        return False

def reconnect_mt5():
    global mt5_initialized, error_count
    try:
        mt5.shutdown()
        mt5_initialized = False
        for attempt in range(CONFIG["MAX_RETRIES"]):
            logger.info(f"Reconnecting MT5, attempt {attempt + 1}", extra=trade_id_context)
            if initialize_mt5():
                error_count = 0
                return True
            time.sleep(CONFIG["RETRY_BACKOFF_FACTOR"] ** attempt)
        logger.error("Failed to reconnect MT5 after all attempts.", extra=trade_id_context)
        error_count += 1
        return False
    except Exception as e:
        logger.error(f"MT5 reconnect error: {str(e)}", extra=trade_id_context)
        error_count += 1
        return False

def fetch_data(symbol, timeframe, bars=CONFIG["MIN_BARS"]):
    try:
        if not mt5_initialized and not reconnect_mt5():
            logger.error(f"MT5 not connected for {symbol} data fetch", extra=trade_id_context)
            return None
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            logger.error(f"Symbol {symbol} not available or not visible: {mt5.last_error()}", extra=trade_id_context)
            return None
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) < CONFIG["MIN_BARS"]:
            logger.error(f"Fetch failed for {symbol}: {len(rates) if rates is not None else 'None'} bars, needed {CONFIG['MIN_BARS']}", extra=trade_id_context)
            return None
        df = pd.DataFrame(rates)
        logger.info(f"Fetched data for {symbol}: shape={df.shape}, columns={df.columns.tolist()}", extra=trade_id_context)
        required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing columns in data for {symbol}: {df.columns.tolist()}", extra=trade_id_context)
            return None
        df = df[required_cols]
        logger.debug(f"Fetched {len(df)} bars for {symbol} timeframe={timeframe}", extra=trade_id_context)
        return df
    except Exception as e:
        logger.error(f"Fetch data error for {symbol}: {str(e)}, Error={mt5.last_error()}", extra=trade_id_context)
        return None

def technical_analysis(df):
    try:
        if df is None or df.empty or len(df) < CONFIG["MIN_BARS"]:
            logger.warning(f"Invalid data for technical analysis: {len(df) if df is not None else 'None'} bars", extra=trade_id_context)
            return None
        df = df.copy()
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['sma'] = talib.SMA(df['close'], timeperiod=CONFIG["SMA_PERIOD"])
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['volatility'] = df['close'].rolling(window=20).std()
        df['rsi'] = df['rsi'].fillna(50)
        df['macd'] = df['macd'].fillna(0)
        df['macd_signal'] = df['macd_signal'].fillna(0)
        df['atr'] = df['atr'].fillna(CONFIG["DEFAULT_ATR"])
        df['sma'] = df['sma'].fillna(df['close'].mean())
        df['price_change_pct'] = df['price_change_pct'].fillna(0)
        df['volatility'] = df['volatility'].fillna(df['close'].std())
        df = df.dropna().reset_index(drop=True)
        if df.empty or len(df) < CONFIG["MIN_BARS"]:
            logger.warning(f"DataFrame empty or insufficient after technical analysis: {len(df)} bars", extra=trade_id_context)
            return None
        logger.info(f"Technical analysis for {len(df)} bars: RSI={df['rsi'].iloc[-1]:.2f}, MACD={df['macd'].iloc[-1]:.6f}, MACD_signal={df['macd_signal'].iloc[-1]:.6f}, ATR={df['atr'].iloc[-1]:.5f}, SMA={df['sma'].iloc[-1]:.2f}, PriceChangePct={df['price_change_pct'].iloc[-1]:.4f}, Volatility={df['volatility'].iloc[-1]:.4f}", extra=trade_id_context)
        logger.debug(f"Technical analysis completed for {len(df)} bars: {df.columns.tolist()}", extra=trade_id_context)
        return df
    except Exception as e:
        logger.error(f"Technical analysis error: {str(e)}", extra=trade_id_context)
        return None

def detect_hidden_divergence(df, lookback=CONFIG["DIVERGENCE_LOOKBACK"]):
    try:
        if df is None or len(df) < lookback:
            logger.warning(f"Insufficient data for divergence detection: {len(df) if df is not None else 'None'} bars", extra=trade_id_context)
            return None, None
        price = df['close'].iloc[-lookback:].reset_index(drop=True)
        rsi = df['rsi'].iloc[-lookback:].reset_index(drop=True)
        price_lows = (price < price.shift(1)) & (price < price.shift(-1))
        price_highs = (price > price.shift(1)) & (price > price.shift(-1))
        rsi_lows = (rsi < rsi.shift(1)) & (rsi < rsi.shift(-1))
        rsi_highs = (rsi > rsi.shift(1)) & (rsi > rsi.shift(-1))
        price_low_indices = price_lows[price_lows].index.tolist()
        price_high_indices = price_highs[price_highs].index.tolist()
        rsi_low_indices = rsi_lows[rsi_lows].index.tolist()
        rsi_high_indices = rsi_highs[rsi_highs].index.tolist()
        is_uptrend = df['close'].iloc[-1] > df['sma'].iloc[-1]
        is_downtrend = df['close'].iloc[-1] < df['sma'].iloc[-1]
        bullish_divergence = None
        bearish_divergence = None
        logger.debug(f"Divergence detection: PriceLowCount={len(price_low_indices)}, PriceHighCount={len(price_high_indices)}, RSILowCount={len(rsi_low_indices)}, RSIHighCount={len(rsi_high_indices)}, Uptrend={is_uptrend}, Downtrend={is_downtrend}", extra=trade_id_context)
        if len(price_low_indices) >= 2 and len(rsi_low_indices) >= 2:
            price_low_idx = price_low_indices[-2:]
            rsi_low_idx = rsi_low_indices[-2:]
            if (is_uptrend and
                price.iloc[price_low_idx[-1]] > price.iloc[price_low_idx[-2]] and
                rsi.iloc[rsi_low_idx[-1]] < rsi.iloc[rsi_low_idx[-2]]):
                bullish_divergence = True
                logger.info(f"Hidden bullish divergence detected: Price Low 1={price.iloc[price_low_idx[-2]]:.2f}, Price Low 2={price.iloc[price_low_idx[-1]]:.2f}, RSI Low 1={rsi.iloc[rsi_low_idx[-2]]:.2f}, RSI Low 2={rsi.iloc[rsi_low_idx[-1]]:.2f}", extra=trade_id_context)
            else:
                bullish_divergence = False
        if len(price_high_indices) >= 2 and len(rsi_high_indices) >= 2:
            price_high_idx = price_high_indices[-2:]
            rsi_high_idx = rsi_high_indices[-2:]
            if (is_downtrend and
                price.iloc[price_high_idx[-1]] < price.iloc[price_high_idx[-2]] and
                rsi.iloc[rsi_high_idx[-1]] > rsi.iloc[rsi_high_idx[-2]]):
                bearish_divergence = True
                logger.info(f"Hidden bearish divergence detected: Price High 1={price.iloc[price_high_idx[-2]]:.2f}, Price High 2={price.iloc[price_high_idx[-1]]:.2f}, RSI High 1={rsi.iloc[rsi_high_idx[-2]]:.2f}, RSI High 2={rsi.iloc[rsi_high_idx[-1]]:.2f}", extra=trade_id_context)
            else:
                bearish_divergence = False
        logger.debug(f"Divergence detection for {df['symbol'].iloc[0] if 'symbol' in df.columns else 'XAUUSD'}: BullishDiv={bullish_divergence}, BearishDiv={bearish_divergence}", extra=trade_id_context)
        return bullish_divergence, bearish_divergence
    except Exception as e:
        logger.error(f"Hidden divergence detection error: {str(e)}", extra=trade_id_context)
        return None, None

def prepare_ml_features(df, open_positions=None):
    try:
        if df is None or df.empty:
            logger.warning("No data for ML features", extra=trade_id_context)
            return None
        features = CONFIG["ML_MODEL_FEATURES"]
        if not all(col in df.columns for col in features):
            logger.error(f"Missing feature columns for ML: {features}, Available: {df.columns.tolist()}", extra=trade_id_context)
            return None
        X = df[features].copy()
        # Add open position features if available
        if open_positions:
            position_data = []
            for pos in open_positions:
                try:
                    profit_percent = (pos.profit / CONFIG["INITIALEQUITY"]) * 100
                    price_change = ((pos.price_current - pos.price_open) / pos.price_open) * 100 if pos.price_open != 0 else 0
                    position_data.append({
                        'rsi': df['rsi'].iloc[-1],
                        'macd': df['macd'].iloc[-1],
                        'macd_signal': df['macd_signal'].iloc[-1],
                        'atr': df['atr'].iloc[-1],
                        'price_change_pct': price_change,
                        'volatility': df['volatility'].iloc[-1],
                        'profit_percent': profit_percent
                    })
                    logger.debug(f"Added position data for ticket {pos.ticket}: PriceOpen={pos.price_open:.5f}, PriceCurrent={pos.price_current:.5f}, ProfitPercent={profit_percent:.2f}%", extra=trade_id_context)
                except AttributeError as e:
                    logger.warning(f"Skipping position due to missing attributes: {str(e)}", extra=trade_id_context)
                    continue
            if position_data:
                pos_df = pd.DataFrame(position_data)
                X = pd.concat([X, pos_df], ignore_index=True)
        X.replace([np.inf, -np.inf], 0, inplace=True)
        X.fillna(0, inplace=True)
        logger.debug(f"Prepared ML features: {X.columns.tolist()}, shape={X.shape}", extra=trade_id_context)
        return X
    except Exception as e:
        logger.error(f"Prepare ML features error: {str(e)}", extra=trade_id_context)
        return None

def train_ml_model(symbol, X_train, y_train):
    global ml_models, ml_trained
    trade_id_context_local = {'trade_id': str(uuid4())}
    try:
        if X_train is None or y_train is None or X_train.empty or y_train.empty:
            logger.warning(f"No valid training data for {symbol}", extra=trade_id_context_local)
            return False
        unique_classes = np.unique(y_train)
        logger.debug(f"Raw class distribution for {symbol}: {dict(zip(unique_classes, np.bincount(y_train + 1)))}", extra=trade_id_context_local)
        if len(unique_classes) < 2:
            logger.warning(f"Cannot train model for {symbol}: Only one class found in raw y: {unique_classes}", extra=trade_id_context_local)
            return False
        X_train = X_train.copy()
        X_train.replace([np.inf, -np.inf], 0, inplace=True)
        X_train.fillna(0, inplace=True)
        y_train = y_train.copy().replace([np.inf, -np.inf], 0).fillna(0)
        smote = SMOTE(random_state=CONFIG["ML_MODEL_SMOTE_RANDOM_STATE"], k_neighbors=CONFIG["ML_MODEL_SMOTE_K_NEIGHBORS"])
        try:
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            logger.debug(f"After SMOTE, class distribution for {symbol}: {dict(zip(np.unique(y_train_balanced), np.bincount(y_train_balanced + 1)))}", extra=trade_id_context_local)
        except ValueError as e:
            logger.warning(f"SMOTE failed for {symbol}: {str(e)}. Proceeding with original data.", extra=trade_id_context_local)
            X_train_balanced, y_train_balanced = X_train, y_train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_balanced)
        model = XGBClassifier(**CONFIG["ML_MODEL_XGB_PARAMS"])
        model.fit(X_scaled, y_train_balanced)
        ml_models[symbol]['primary'] = {'model': model, 'scaler': scaler}
        ml_trained[symbol]['primary'] = True
        save_ml_model(symbol)
        logger.info(f"ML model trained for {symbol} with {len(X_train_balanced)} samples", extra=trade_id_context_local)
        return True
    except Exception as e:
        logger.error(f"Train ML model error for {symbol}: {str(e)}", extra=trade_id_context_local)
        return False

def update_ml_model(symbol):
    global last_ml_train
    trade_id_context_local = {'trade_id': str(uuid4())}
    try:
        # Fetch market data
        df = fetch_data(symbol, CONFIG["TIMEFRAMES"]["M5"], bars=CONFIG["MIN_BARS"])
        if df is None or df.empty:
            logger.warning(f"No data for ML update for {symbol}", extra=trade_id_context_local)
            return False
        df = technical_analysis(df)
        if df is None or df.empty:
            logger.warning(f"Technical analysis failed for ML update for {symbol}", extra=trade_id_context_local)
            return False
        # Fetch open positions
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions:
            logger.debug(f"Open positions for {symbol}: {len(open_positions)}", extra=trade_id_context_local)
        X = prepare_ml_features(df, open_positions)
        if X is None or X.empty:
            logger.warning(f"No features for ML update for {symbol}", extra=trade_id_context_local)
            return False
        # Calculate target variable
        price_change = df['close'].shift(-1) / df['close'] - 1
        logger.debug(f"Price change distribution: {price_change.describe().to_dict()}", extra=trade_id_context_local)
        threshold = max(CONFIG["PRICE_CHANGE_THRESHOLD"], price_change.std() * 0.2)
        logger.debug(f"Dynamic threshold for {symbol}: {threshold:.6f}", extra=trade_id_context_local)
        y = pd.Series(0, index=X.index[:len(df)])
        y[price_change > threshold] = 1
        y[price_change < -threshold] = -1
        # Add open position outcomes
        if open_positions:
            for pos in open_positions:
                try:
                    profit_percent = (pos.profit / CONFIG["INITIALEQUITY"]) * 100
                    y = pd.concat([y, pd.Series(1 if profit_percent > CONFIG["MIN_PROFIT_PERCENT"] else -1 if profit_percent < -CONFIG["MIN_PROFIT_PERCENT"] else 0)], ignore_index=True)
                    logger.debug(f"Position outcome for ticket {pos.ticket}: ProfitPercent={profit_percent:.2f}%", extra=trade_id_context_local)
                except AttributeError as e:
                    logger.warning(f"Skipping position due to missing attributes: {str(e)}", extra=trade_id_context_local)
                    continue
        logger.debug(f"Raw y distribution: {dict(zip(np.unique(y), np.bincount(y + 1)))}", extra=trade_id_context_local)
        y_mapped = y.copy().replace(-1, 0).replace(0, 1).replace(1, 2)
        logger.debug(f"Mapped y distribution: {dict(zip(np.unique(y_mapped), np.bincount(y_mapped.astype(int))))}", extra=trade_id_context_local)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        valid_idx = X.index.intersection(y.index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        logger.debug(f"X shape: {X.shape}, y shape: {y.shape}, valid_idx length: {len(valid_idx)}", extra=trade_id_context_local)
        if len(X) < CONFIG["MIN_BARS"]:
            logger.warning(f"Insufficient data for ML training for {symbol}: {len(X)} samples", extra=trade_id_context_local)
            return False
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning(f"Skipping ML training for {symbol}: Only one class found: {unique_classes}", extra=trade_id_context_local)
            return False
        success = train_ml_model(symbol, X.iloc[:-10], y.iloc[:-10])
        if success:
            last_ml_train[symbol]['primary'] = datetime.now()
        else:
            logger.warning(f"ML training failed for {symbol}. Falling back to rule-based trading.", extra=trade_id_context_local)
            ml_trained[symbol]['primary'] = False
        return success
    except Exception as e:
        logger.error(f"Update ML error for {symbol}: {str(e)}", extra=trade_id_context_local)
        return False

def trading_decision(symbol, df):
    trade_id = str(uuid4())
    trade_id_context['trade_id'] = trade_id
    try:
        if not mt5_initialized and not reconnect_mt5():
            logger.error(f"MT5 not initialized for {symbol}", extra=trade_id_context)
            return None, None, None, 0.0, "None", trade_id
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            logger.error(f"Symbol info not found or not visible for {symbol}: {mt5.last_error()}", extra=trade_id_context)
            return None, None, None, 0.0, "None", trade_id
        # Retry fetching tick data
        for attempt in range(CONFIG["MAX_RETRIES"]):
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None and tick.bid > 0 and tick.ask > 0:
                break
            logger.warning(f"Invalid tick data for {symbol} on attempt {attempt + 1}: Bid={tick.bid if tick else 'None'}, Ask={tick.ask if tick else 'None'}", extra=trade_id_context)
            time.sleep(CONFIG["RETRY_BACKOFF_FACTOR"] ** attempt)
        else:
            logger.error(f"Failed to fetch valid tick data for {symbol} after {CONFIG['MAX_RETRIES']} attempts", extra=trade_id_context)
            return None, None, None, 0.0, "None", trade_id
        if df is None or df.empty:
            logger.debug(f"No data for {symbol}", extra=trade_id_context)
            return None, None, None, 0.0, "None", trade_id
        latest = df.iloc[-1]
        signal_strength = 0.0
        strategy_type = "rule-based"
        bullish_divergence, bearish_divergence = detect_hidden_divergence(df)
        is_uptrend = latest['close'] > latest['sma']
        is_downtrend = latest['close'] < latest['sma']
        macd_above_signal = latest['macd'] > latest['macd_signal']
        macd_below_signal = latest['macd'] < latest['macd_signal']
        logger.info(f"Evaluating rule-based for {symbol}: RSI={latest['rsi']:.2f}, BullishDiv={bullish_divergence}, BearishDiv={bearish_divergence}, Close={latest['close']:.2f}, SMA={latest['sma']:.2f}, Uptrend={is_uptrend}, Downtrend={is_downtrend}, MACD={latest['macd']:.6f}, MACD_Signal={latest['macd_signal']:.6f}, MACD_Above={macd_above_signal}, MACD_Below={macd_below_signal}", extra=trade_id_context)
        
        # Rule-based trading logic
        buy_conditions = [
            latest['rsi'] < CONFIG["RSI_BUY_THRESHOLD"],
            bullish_divergence == True,
            macd_above_signal and is_uptrend
        ]
        sell_conditions = [
            latest['rsi'] > CONFIG["RSI_SELL_THRESHOLD"],
            bearish_divergence == True,
            macd_below_signal
        ]
        buy_score = sum(1 for cond in buy_conditions if cond)
        sell_score = sum(1 for cond in sell_conditions if cond)
        logger.debug(f"Signal scores for {symbol}: BuyConditions={buy_conditions}, SellConditions={sell_conditions}, BuyScore={buy_score}, SellScore={sell_score}", extra=trade_id_context)
        
        # Prioritize stronger signal
        if buy_score > sell_score and buy_score >= 1:
            signal_strength = 0.5 * buy_score
            action = "buy"
            strategy_type = "rsi-divergence-macd-trend" if (bullish_divergence or macd_above_signal) else "rsi-based"
            logger.info(f"Buy signal for {symbol}: RSI={latest['rsi']:.2f}, BullishDiv={bullish_divergence}, MACD_Above={macd_above_signal}, Uptrend={is_uptrend}, Strategy={strategy_type}, BuyScore={buy_score}", extra=trade_id_context)
        elif sell_score >= 1 and latest['rsi'] > 70:
            signal_strength = -0.5 * sell_score
            action = "sell"
            strategy_type = "rsi-based"
            logger.info(f"Sell signal for {symbol}: RSI={latest['rsi']:.2f}, Strategy={strategy_type}, SellScore={sell_score}", extra=trade_id_context)
        elif sell_score > buy_score and sell_score >= 1 and (is_downtrend or (macd_below_signal and latest['rsi'] > 50)):
            signal_strength = -0.5 * sell_score
            action = "sell"
            strategy_type = "rsi-divergence-macd-trend" if (bearish_divergence or macd_below_signal) else "rsi-based"
            logger.info(f"Sell signal for {symbol}: RSI={latest['rsi']:.2f}, BearishDiv={bearish_divergence}, MACD_Below={macd_below_signal}, Downtrend={is_downtrend}, Strategy={strategy_type}, SellScore={sell_score}", extra=trade_id_context)
        elif buy_score == sell_score and buy_score >= 1:
            if is_uptrend:
                signal_strength = 0.3
                action = "buy"
                strategy_type = "trend-based"
                logger.info(f"Buy signal for {symbol}: Uptrend={is_uptrend}, Strategy={strategy_type}, BuyScore={buy_score}, SellScore={sell_score}", extra=trade_id_context)
            elif is_downtrend:
                signal_strength = -0.3
                action = "sell"
                strategy_type = "trend-based"
                logger.info(f"Sell signal for {symbol}: Downtrend={is_downtrend}, Strategy={strategy_type}, BuyScore={buy_score}, SellScore={sell_score}", extra=trade_id_context)
            else:
                action = None
                logger.info(f"No signal for {symbol}: Neutral trend, BuyScore={buy_score}, SellScore={sell_score}, RSI={latest['rsi']:.2f}, BullishDiv={bullish_divergence}, BearishDiv={bearish_divergence}, MACD_Above={macd_above_signal}, MACD_Below={macd_below_signal}", extra=trade_id_context)
        else:
            action = None
            logger.info(f"No signal for {symbol}: BuyScore={buy_score}, SellScore={sell_score}, RSI={latest['rsi']:.2f}, BullishDiv={bullish_divergence}, BearishDiv={bearish_divergence}, MACD_Above={macd_above_signal}, MACD_Below={macd_below_signal}", extra=trade_id_context)
        
        # ML-based trading if enabled
        if not CONFIG["FORCE_RULE_BASED"] and ml_trained[symbol]['primary']:
            features = CONFIG["ML_MODEL_FEATURES"]
            if all(col in df.columns for col in features):
                X = pd.DataFrame([latest[features]], columns=features)
                X.replace([np.inf, -np.inf], 0, inplace=True)
                X.fillna(0, inplace=True)
                X_scaled = ml_models[symbol]['primary']['scaler'].transform(X)
                pred = ml_models[symbol]['primary']['model'].predict_proba(X_scaled)[0]
                logger.debug(f"ML prediction for {symbol}: Sell={pred[0]:.2f}, Neutral={pred[1]:.2f}, Buy={pred[2]:.2f}", extra=trade_id_context)
                if pred[2] > 0.7:
                    signal_strength = max(signal_strength, 0.5)
                    action = "buy"
                    strategy_type = "ml-based"
                elif pred[0] > 0.7:
                    signal_strength = min(signal_strength, -0.5)
                    action = "sell"
                    strategy_type = "ml-based"
        
        if action and abs(signal_strength) >= CONFIG["SIGNAL_THRESHOLD"]:
            atr = latest['atr'] if 'atr' in df.columns else CONFIG["DEFAULT_ATR"]
            current_price = tick.ask if action == "buy" else tick.bid
            point = symbol_info.point
            min_stop_distance = symbol_info.trade_stops_level * point
            sl_distance = max(atr * CONFIG["MIN_STOP_DISTANCE_FACTOR"], min_stop_distance)
            tp_distance = sl_distance * 2.0
            sl_price = current_price - sl_distance if action == "buy" else current_price + sl_distance
            tp_price = current_price + tp_distance if action == "buy" else current_price - tp_distance
            logger.info(f"Trade signal for {symbol}: Action={action}, Signal={signal_strength:.2f}, Price={current_price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}, Strategy={strategy_type}, ATR={atr:.5f}, MinStopDistance={min_stop_distance:.5f}", extra=trade_id_context)
            return action, sl_price, tp_price, signal_strength, strategy_type, trade_id
        logger.info(f"No trade for {symbol}: Signal strength={signal_strength:.2f}, Threshold={CONFIG['SIGNAL_THRESHOLD']}", extra=trade_id_context)
        return None, None, None, 0.0, "None", trade_id
    except Exception as e:
        logger.error(f"Trading decision error for {symbol}: {str(e)}", extra=trade_id_context)
        return None, None, None, 0.0, "None", trade_id

def calculate_volume(symbol, sl_distance):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"No symbol info for {symbol}. Using minimum lot size.", extra=trade_id_context)
            return CONFIG["MIN_LOT_SIZE"]
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"No account info available. Using minimum lot size.", extra=trade_id_context)
            return CONFIG["MIN_LOT_SIZE"]
        equity = account_info.equity
        risk_amount = equity * (CONFIG["RISK_PER_TRADE_PERCENT"] / 100)
        point = symbol_info.point
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        if tick_size == 0 or tick_value == 0 or sl_distance == 0:
            logger.error(f"Invalid tick data for {symbol}: TickSize={tick_size}, TickValue={tick_value}, SLDistance={sl_distance}. Using minimum lot size.", extra=trade_id_context)
            return CONFIG["MIN_LOT_SIZE"]
        sl_points = abs(sl_distance / tick_size)
        volume = risk_amount / (sl_points * tick_value)
        volume_step = symbol_info.volume_step or 0.01
        volume = round(volume / volume_step) * volume_step
        volume = max(symbol_info.volume_min, min(CONFIG["MAX_LOT_SIZE"], volume))
        logger.info(f"Volume calculation for {symbol}: Volume={volume}, Equity={equity}, Risk={risk_amount}, SLDistance={sl_distance}, SLPoints={sl_points}, TickValue={tick_value}, VolumeMin={symbol_info.volume_min}, VolumeMax={symbol_info.volume_max}", extra=trade_id_context)
        return volume
    except Exception as e:
        logger.error(f"Volume calculation error for {symbol}: {str(e)}", extra=trade_id_context)
        return CONFIG["MIN_LOT_SIZE"]

def validate_trade_request(symbol, action, price, sl_price, tp_price, volume):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found for {symbol}", extra=trade_id_context)
            return False
        point = symbol_info.point
        min_stop_distance = symbol_info.trade_stops_level * point
        digits = symbol_info.digits
        price = round(price, digits)
        sl_price = round(sl_price, digits) if sl_price else 0.0
        tp_price = round(tp_price, digits) if tp_price else 0.0
        sl_distance = abs(price - sl_price) if sl_price else 0.0
        if sl_distance > 0 and sl_distance < min_stop_distance:
            logger.error(f"Stop loss too close for {symbol}: Distance={sl_distance}, Min={min_stop_distance}", extra=trade_id_context)
            return False
        if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
            logger.error(f"Invalid volume for {symbol}: Volume={volume}, Min={symbol_info.volume_min}, Max={symbol_info.volume_max}", extra=trade_id_context)
            return False
        logger.debug(f"Trade request validated for {symbol}: Price={price}, SL={sl_price}, TP={tp_price}, Volume={volume}, MinStopDistance={min_stop_distance}", extra=trade_id_context)
        return True
    except Exception as e:
        logger.error(f"Trade request validation error for {symbol}: {str(e)}", extra=trade_id_context)
        return False

def execute_trade(symbol, action, sl_price, tp_price, signal_strength, strategy_type, trade_id):
    trade_id_context['trade_id'] = trade_id
    try:
        if CONFIG["DIAGNOSTIC_MODE"]:
            volume = calculate_volume(symbol, abs(sl_price - (mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid)))
            logger.info(f"DIAGNOSTIC MODE: Would execute trade for {symbol}: Action={action}, SL={sl_price:.5f}, TP={tp_price:.5f}, Volume={volume}, Strategy={strategy_type}", extra=trade_id_context)
            return None
        if not mt5_initialized and not reconnect_mt5():
            logger.error(f"MT5 not connected for trade execution for {symbol}", extra=trade_id_context)
            return None
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found for {symbol}: {mt5.last_error()}", extra=trade_id_context)
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            logger.error(f"Invalid tick data for {symbol}: Bid={tick.bid if tick else 'None'}, Ask={tick.ask if tick else 'None'}", extra=trade_id_context)
            return None
        price = tick.ask if action == "buy" else tick.bid
        digits = symbol_info.digits
        price = round(price, digits)
        sl_price = round(sl_price, digits) if sl_price else 0.0
        tp_price = round(tp_price, digits) if tp_price else 0.0
        sl_distance = abs(price - sl_price)
        volume = calculate_volume(symbol, sl_distance)
        logger.info(f"Preparing trade for {symbol}: Action={action}, Price={price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}, Volume={volume}, Strategy={strategy_type}", extra=trade_id_context)
        if not validate_trade_request(symbol, action, price, sl_price, tp_price, volume):
            logger.error(f"Trade request validation failed for {symbol}", extra=trade_id_context)
            return None
        open_positions = mt5.positions_get(symbol=symbol)
        if open_positions and len(open_positions) >= CONFIG["MAX_OPEN_TRADES_PER_SYMBOL"]:
            logger.warning(f"Max open trades reached for {symbol}: {len(open_positions)}", extra=trade_id_context)
            return None
        comment = f"{action.upper()}_RSI_MACD_{trade_id[:8]}"
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": comment
        }
        for attempt in range(CONFIG["MAX_RETRIES"]):
            logger.info(f"Sending trade request for {symbol} (Attempt {attempt + 1}): Action={action}, Price={price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}, Volume={volume}, Comment={comment}", extra=trade_id_context)
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"Order send failed for {symbol}: MT5 returned None, Error={mt5.last_error()}", extra=trade_id_context)
                time.sleep(CONFIG["RETRY_BACKOFF_FACTOR"] ** attempt)
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                position = type('Position', (), {
                    'symbol': symbol,
                    'action': action,
                    'price_open': float(price),
                    'volume': float(volume),
                    'sl': float(sl_price),
                    'tp': float(tp_price),
                    'ticket': result.order,
                    'signal_strength': signal_strength,
                    'open_time': datetime.now(),
                    'strategy_type': strategy_type,
                    'trade_id': trade_id
                })()
                monitored_positions.append(position)
                logger.info(f"Trade opened for {symbol}: {action.upper()}, Ticket={position.ticket}, Price={price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}, Volume={volume}", extra=trade_id_context)
                trade_history.append({
                    "trade_id": trade_id,
                    "timestamp": str(datetime.now()),
                    "symbol": symbol,
                    "action": action,
                    "price_open": price,
                    "volume": volume,
                    "sl": sl_price,
                    "tp": tp_price,
                    "ticket": str(result.order),
                    "signal_strength": signal_strength,
                    "strategy_type": strategy_type
                })
                save_state()
                return position
            else:
                logger.error(f"Trade failed for {symbol}: Retcode={result.retcode}, Comment={result.comment}, Error={mt5.last_error()}", extra=trade_id_context)
                time.sleep(CONFIG["RETRY_BACKOFF_FACTOR"] ** attempt)
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logger.error(f"Failed to get updated tick data for {symbol}", extra=trade_id_context)
                    return None
                price = tick.ask if action == "buy" else tick.bid
                price = round(price, digits)
                request["price"] = float(price)
                continue
        logger.error(f"Trade failed for {symbol} after {CONFIG['MAX_RETRIES']} attempts", extra=trade_id_context)
        return None
    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {str(e)}, Request={request}", extra=trade_id_context)
        return None

async def monitor_trades(symbol):
    try:
        while True:
            try:
                if not mt5_initialized and not reconnect_mt5():
                    logger.error(f"MT5 not connected for monitoring {symbol}", extra=trade_id_context)
                    await asyncio.sleep(1)
                    continue
                open_positions = mt5.positions_get(symbol=symbol)
                open_trades_count = len(open_positions) if open_positions else 0
                existing_tickets = {pos.ticket for pos in monitored_positions if pos.symbol == symbol}
                for pos in open_positions if open_positions else []:
                    if pos.ticket not in existing_tickets:
                        position = type('Position', (), {
                            'symbol': symbol,
                            'action': 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                            'price_open': float(pos.price_open),
                            'volume': float(pos.volume),
                            'sl': float(pos.sl),
                            'tp': float(pos.tp),
                            'ticket': pos.ticket,
                            'signal_strength': 0.5,
                            'open_time': datetime.fromtimestamp(pos.time),
                            'strategy_type': 'external',
                            'trade_id': str(uuid4())
                        })()
                        monitored_positions.append(position)
                        logger.info(f"Added externally opened position for {symbol}: Ticket={pos.ticket}, Action={position.action}, Price={pos.price_open:.5f}, SL={pos.sl:.5f}, TP={pos.tp:.5f}", extra=trade_id_context)
                symbol_positions = [pos for pos in monitored_positions if pos.symbol == symbol]
                if symbol_positions:
                    position_details = [
                        f"Ticket={pos.ticket}, Action={pos.action.upper()}, Price={pos.price_open:.5f}, SL={pos.sl:.5f}, TP={pos.tp:.5f}, Strategy={pos.strategy_type}"
                        for pos in symbol_positions
                    ]
                    logger.info(f"Open positions for {symbol}: {len(symbol_positions)}/{CONFIG['MAX_OPEN_TRADES_PER_SYMBOL']} - {position_details}", extra=trade_id_context)
                else:
                    logger.info(f"No open positions for {symbol}", extra=trade_id_context)
                for position in symbol_positions[:]:
                    trade_id_context['trade_id'] = position.trade_id
                    positions = mt5.positions_get(ticket=position.ticket)
                    if not positions:
                        logger.warning(f"Position {position.ticket} for {symbol} closed externally", extra=trade_id_context)
                        if position in monitored_positions:
                            monitored_positions.remove(position)
                        continue
                    pos = positions[0]
                    current_price = pos.price_current
                    profit = float(pos.profit)
                    profit_percent = (profit / CONFIG["INITIALEQUITY"]) * 100
                    logger.debug(f"Monitoring {symbol} position {pos.ticket}: CurrentPrice={current_price:.5f}, ProfitPercent={profit_percent:.2f}%", extra=trade_id_context)
                    sl_triggered = (current_price <= pos.sl and position.action == "buy") or (current_price >= pos.sl and position.action == "sell")
                    tp_triggered = (current_price >= pos.tp and position.action == "buy") or (current_price <= pos.tp and position.action == "sell")
                    if sl_triggered:
                        logger.info(f"Closing {symbol} trade {position.ticket} due to stop loss: Profit={profit_percent:.2f}%", extra=trade_id_context)
                        close_trade(position, "Stop Loss", profit, current_price)
                    elif tp_triggered:
                        logger.info(f"Closing {symbol} trade {position.ticket} due to take profit: Profit={profit_percent:.2f}%", extra=trade_id_context)
                        close_trade(position, "Take Profit", profit, current_price)
                    elif profit_percent >= CONFIG["TARGET_PROFIT_PERCENT"]:
                        logger.info(f"Closing {symbol} trade {position.ticket} for target profit: {profit_percent:.2f}%", extra=trade_id_context)
                        close_trade(position, "Profit Target", profit, current_price)
                    elif profit_percent >= CONFIG["MIN_PROFIT_PERCENT"]:
                        logger.info(f"Closing {symbol} trade {position.ticket} for minimal profit: {profit_percent:.2f}%", extra=trade_id_context)
                        close_trade(position, "Minimal Profit", profit, current_price)
                    elif profit_percent <= -CONFIG["MIN_PROFIT_PERCENT"]:
                        logger.info(f"Closing {symbol} trade {position.ticket} for max loss: {profit_percent:.2f}%", extra=trade_id_context)
                        close_trade(position, "Max Loss", profit, current_price)
            except Exception as e:
                logger.error(f"Error in monitor loop for {symbol}: {str(e)}", extra=trade_id_context)
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Fatal error in monitor_trades for {symbol}: {str(e)}", extra=trade_id_context)

def close_trade(position, reason, profit, price_close):
    global trade_stats
    trade_id_context['trade_id'] = position.trade_id
    try:
        if CONFIG["DIAGNOSTIC_MODE"]:
            logger.info(f"DIAGNOSTIC MODE: Would close trade for {position.symbol}: Ticket={position.ticket}, Reason={reason}, Profit={profit:.2f}, ClosePrice={price_close:.5f}", extra=trade_id_context)
            if position in monitored_positions:
                monitored_positions.remove(position)
            return True
        if not mt5_initialized and not reconnect_mt5():
            logger.error(f"MT5 not connected for closing position, Ticket={position.ticket}", extra=trade_id_context)
            return False
        symbol_info = mt5.symbol_info(position.symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found for {position.symbol}: {mt5.last_error()}", extra=trade_id_context)
            return False
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            logger.error(f"Tick data not found for {position.symbol}: {mt5.last_error()}", extra=trade_id_context)
            return False
        price = tick.bid if position.action == "buy" else tick.ask
        digits = symbol_info.digits
        price = round(price, digits)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": mt5.ORDER_TYPE_SELL if position.action == "buy" else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": float(price),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "comment": f"Close {reason}"
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close trade, Ticket={position.ticket}: Retcode={result.retcode}, Comment={result.comment}, Error={mt5.last_error()}", extra=trade_id_context)
            return False
        trade_stats[position.symbol]["total_trades"] += 1
        if profit > 0:
            trade_stats[position.symbol]["wins"] += 1
        else:
            trade_stats[position.symbol]["losses"] += 1
        trade_stats[position.symbol]["net_profit"] += profit
        trade_history.append({
            "trade_id": position.trade_id,
            "timestamp": str(datetime.now()),
            "symbol": position.symbol,
            "action": "close",
            "ticket": str(position.ticket),
            "profit": float(profit),
            "reason": reason,
            "price_close": float(price_close),
            "strategy_type": position.strategy_type
        })
        if position in monitored_positions:
            monitored_positions.remove(position)
        logger.info(f"Trade closed for {position.symbol}: Ticket={position.ticket}, Profit={profit:.2f}, Reason={reason}, ClosePrice={price_close:.5f}", extra=trade_id_context)
        save_state()
        return True
    except Exception as e:
        logger.error(f"Error closing trade, Ticket={position.ticket}: {str(e)}", extra=trade_id_context)
        return False

async def test_credentials():
    try:
        logger.info(f"Testing credentials with server: {CONFIG['MT5_SERVER']}", extra=trade_id_context)
        if mt5.initialize(
            login=CONFIG["MT5_LOGIN"],
            password=CONFIG["MT5_PASSWORD"],
            server=CONFIG["MT5_SERVER"],
            timeout=60000
        ):
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Credentials valid: Balance={account_info.balance}, Equity={account_info.equity}", extra=trade_id_context)
                mt5.shutdown()
                return True
            else:
                logger.error(f"Failed to retrieve account info", extra=trade_id_context)
        error = mt5.last_error()
        logger.error(f"Credential test failed: Code={error[0]}, Message={error[1]}", extra=trade_id_context)
        mt5.shutdown()
        return False
    except Exception as e:
        logger.error(f"Error testing credentials: {str(e)}", extra=trade_id_context)
        return False

async def main_trading():
    global error_count
    init_db()
    try:
        logger.info("Starting trading...", extra=trade_id_context)
        if not await test_credentials():
            logger.error("Invalid credentials. Update CONFIG with correct MT5_LOGIN, MT5_PASSWORD, MT5_SERVER.", extra=trade_id_context)
            sys.exit(1)
        if not initialize_mt5():
            logger.error("Failed to initialize MT5. Verify credentials and MT5 terminal.", extra=trade_id_context)
            sys.exit(1)
        for symbol in CONFIG["SYMBOLS"]:
            if not CONFIG["FORCE_RULE_BASED"]:
                if not load_ml_model(symbol):
                    logger.info(f"Training new ML model for {symbol}", extra=trade_id_context)
                    update_ml_model(symbol)
            else:
                logger.info(f"Forcing rule-based trading for {symbol}", extra=trade_id_context)
                ml_trained[symbol]['primary'] = False
            asyncio.create_task(monitor_trades(symbol))
        while True:
            try:
                if error_count >= CONFIG["MAX_ERRORS"]:
                    logger.error("Maximum errors reached. Terminating...", extra=trade_id_context)
                    break
                if not mt5_initialized and not reconnect_mt5():
                    logger.warning("MT5 not connected, retrying...", extra=trade_id_context)
                    await asyncio.sleep(5)
                    continue
                for symbol in CONFIG["SYMBOLS"]:
                    symbol_info = mt5.symbol_info(symbol)
                    if not symbol_info or not symbol_info.visible:
                        logger.warning(f"Skipping {symbol}: Symbol not available", extra=trade_id_context)
                        await asyncio.sleep(1)
                        continue
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick or tick.bid <= 0 or tick.ask <= 0:
                        logger.warning(f"Skipping {symbol}: Market closed or invalid tick data", extra=trade_id_context)
                        await asyncio.sleep(1)
                        continue
                    # Log ML model status
                    logger.info(f"ML model status for {symbol}: Trained={ml_trained[symbol]['primary']}, Last Trained={last_ml_train[symbol]['primary']}", extra=trade_id_context)
                    # Force ML update regardless of open trades
                    if not CONFIG["FORCE_RULE_BASED"] and datetime.now() - last_ml_train[symbol]['primary'] > CONFIG["MLTRAIN_INTERVAL"]:
                        logger.info(f"Forcing ML model update for {symbol}", extra=trade_id_context)
                        update_ml_model(symbol)
                    open_positions = mt5.positions_get(symbol=symbol)
                    open_trades_count = len(open_positions) if open_positions else 0
                    if open_trades_count >= CONFIG["MAX_OPEN_TRADES_PER_SYMBOL"]:
                        logger.info(f"Max open trades ({open_trades_count}/{CONFIG['MAX_OPEN_TRADES_PER_SYMBOL']}) reached for {symbol}. Monitoring positions only.", extra=trade_id_context)
                        await asyncio.sleep(1)
                        continue
                    logger.info(f"Open trades for {symbol}: {open_trades_count}/{CONFIG['MAX_OPEN_TRADES_PER_SYMBOL']}. Evaluating trading conditions.", extra=trade_id_context)
                    df = fetch_data(symbol, CONFIG["TIMEFRAMES"]["M5"])
                    if df is not None and not df.empty:
                        df = technical_analysis(df)
                        if df is not None and not df.empty:
                            action, sl_price, tp_price, signal_strength, strategy_type, trade_id = trading_decision(symbol, df)
                            trade_id_context['trade_id'] = trade_id
                            if action:
                                logger.info(f"Attempting to execute trade for {symbol}: Action={action}, Strategy={strategy_type}", extra=trade_id_context)
                                result = execute_trade(symbol, action, sl_price, tp_price, signal_strength, strategy_type, trade_id)
                                if result:
                                    logger.info(f"Trade executed for {symbol}: Ticket={result.ticket}", extra=trade_id_context)
                                elif CONFIG["DIAGNOSTIC_MODE"]:
                                    logger.info(f"Trade not executed for {symbol} due to DIAGNOSTIC_MODE", extra=trade_id_context)
                                else:
                                    logger.warning(f"Trade execution failed for {symbol}. Check MT5 error logs.", extra=trade_id_context)
                            else:
                                logger.debug(f"No trade signal for {symbol}", extra=trade_id_context)
                    else:
                        logger.warning(f"No data for {symbol}", extra=trade_id_context)
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Trading loop error: {str(e)}", extra=trade_id_context)
                error_count += 1
                await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Main trading loop error: {str(e)}", extra=trade_id_context)
        graceful_shutdown()
        sys.exit(1)
    finally:
        graceful_shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main_trading())
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}", extra=trade_id_context)
        graceful_shutdown()
        sys.exit(1)
