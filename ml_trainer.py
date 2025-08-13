
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class MLModelTrainer:
    """
    Trains ML models using data collected from the bot's trading activities.
    This allows the bot to learn from its past decisions and improve over time.
    """
    
    def __init__(self, config_file="config.json"):
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        self.spreadsheet_id = config['settings'].get('google_sheet_id')
        
        # Setup Google Sheets connection
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        self.gc = gspread.authorize(creds)
        
        # Define feature columns for training
        self.feature_columns = [
            'current_price', 'volume_24h', 'volatility_24h',
            'pri_rsi', 'pri_ema_fast', 'pri_ema_slow', 'pri_macd', 'pri_macd_signal', 'pri_macd_hist',
            'pri_bb_upper', 'pri_bb_middle', 'pri_bb_lower', 'pri_stoch_rsi_k', 'pri_stoch_rsi_d',
            'pri_adx', 'pri_adx_pos', 'pri_adx_neg',
            'conf_rsi', 'conf_ema_fast', 'conf_ema_slow', 'conf_macd', 'conf_macd_signal', 'conf_macd_hist',
            'conf_bb_upper', 'conf_bb_middle', 'conf_bb_lower', 'conf_stoch_rsi_k', 'conf_stoch_rsi_d',
            'conf_adx', 'conf_adx_pos', 'conf_adx_neg',
            'primary_score', 'confirm_score', 'combined_score',
            'portfolio_balance', 'open_positions_count', 'daily_pnl'
        ]
        
    def load_training_data(self):
        """
        Loads training data from the ML_Training_Data Google Sheet.
        """
        try:
            sheet = self.gc.open_by_key(self.spreadsheet_id)
            ml_worksheet = sheet.worksheet("ML_Training_Data")
            
            # Get all records
            records = ml_worksheet.get_all_records()
            df = pd.DataFrame(records)
            
            # Filter only completed trades (not PENDING)
            df_completed = df[df['outcome_classification'] != 'PENDING'].copy()
            
            logger.info(f"Loaded {len(df_completed)} completed trades for training")
            return df_completed
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df):
        """
        Prepares features for ML training by cleaning and processing the data.
        """
        # Convert numeric columns
        numeric_columns = self.feature_columns + ['outcome_pnl']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create target variable based on outcome classification
        df['target'] = df['outcome_classification'].map({
            'WIN_STRONG': 2,    # Strong positive outcome
            'WIN_SMALL': 1,     # Small positive outcome  
            'LOSS_SMALL': 0,    # Small loss (neutral/avoid)
            'LOSS_STRONG': -1   # Strong negative outcome (avoid)
        })
        
        # For binary classification, convert to profitable (1) vs not profitable (0)
        df['target_binary'] = (df['target'] > 0).astype(int)
        
        # Remove rows with missing target
        df = df.dropna(subset=['target_binary'])
        
        # Select feature columns that exist in the dataframe
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        # Handle missing values
        for col in available_features:
            df[col] = df[col].fillna(df[col].median())
        
        logger.info(f"Prepared {len(df)} samples with {len(available_features)} features")
        return df[available_features + ['target_binary', 'target', 'symbol']], available_features
    
    def train_signal_classifier(self, df, features):
        """
        Trains a binary classifier to predict if a signal will be profitable.
        """
        if len(df) < 50:
            logger.warning(f"Insufficient data for training: {len(df)} samples. Need at least 50.")
            return None
        
        # Prepare features and target
        X = df[features]
        y = df['target_binary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 Most Important Features:\n{feature_importance.head()}")
        
        return model, accuracy, feature_importance
    
    def train_pnl_regressor(self, df, features):
        """
        Trains a regressor to predict the expected PnL of a trade.
        """
        if len(df) < 50:
            logger.warning(f"Insufficient data for PnL regression: {len(df)} samples.")
            return None
        
        # Prepare features and target (PnL)
        X = df[features]
        y = df['outcome_pnl']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        logger.info(f"PnL Regressor trained with RMSE: {rmse:.3f}")
        
        return model, rmse
    
    def save_models(self, classifier, regressor, features, accuracy, rmse):
        """
        Saves the trained models and metadata.
        """
        try:
            # Save classifier
            if classifier:
                joblib.dump(classifier, 'ml_signal_classifier.joblib')
                logger.info("Signal classifier saved")
            
            # Save regressor  
            if regressor:
                joblib.dump(regressor, 'ml_pnl_regressor.joblib')
                logger.info("PnL regressor saved")
            
            # Save metadata
            metadata = {
                'features': features,
                'classifier_accuracy': float(accuracy) if accuracy else None,
                'regressor_rmse': float(rmse) if rmse else None,
                'training_date': pd.Timestamp.now().isoformat(),
                'feature_count': len(features)
            }
            
            with open('ml_model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Model metadata saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def train_all_models(self):
        """
        Main training function that loads data and trains both models.
        """
        logger.info("Starting ML model training...")
        
        # Load training data
        df = self.load_training_data()
        if df.empty:
            logger.error("No training data available")
            return
        
        # Prepare features
        df_prepared, features = self.prepare_features(df)
        if df_prepared.empty:
            logger.error("Failed to prepare features")
            return
        
        # Train classifier
        classifier_result = self.train_signal_classifier(df_prepared, features)
        classifier, accuracy, feature_importance = classifier_result if classifier_result else (None, None, None)
        
        # Train regressor
        regressor_result = self.train_pnl_regressor(df_prepared, features)
        regressor, rmse = regressor_result if regressor_result else (None, None)
        
        # Save models
        self.save_models(classifier, regressor, features, accuracy, rmse)
        
        # Summary
        logger.info("="*50)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Data points used: {len(df_prepared)}")
        logger.info(f"Features used: {len(features)}")
        if accuracy:
            logger.info(f"Signal Classifier Accuracy: {accuracy:.3f}")
        if rmse:
            logger.info(f"PnL Regressor RMSE: {rmse:.3f}")
        logger.info("="*50)

if __name__ == "__main__":
    trainer = MLModelTrainer()
    trainer.train_all_models()
