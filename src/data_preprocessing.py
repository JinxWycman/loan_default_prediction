import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanDataPreprocessor:
    """Complete data preprocessing pipeline for loan default prediction"""
    
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path):
        """Load dataset from CSV"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df):
        """Handle missing values in both numeric and categorical columns"""
        # Create copy to avoid warnings
        df_processed = df.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'Default' in numeric_cols:
            numeric_cols.remove('Default')
        
        logger.info(f"Numeric columns: {len(numeric_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        
        # Impute numeric columns
        if numeric_cols:
            df_processed[numeric_cols] = self.numeric_imputer.fit_transform(df_processed[numeric_cols])
        
        # Impute categorical columns
        if categorical_cols:
            df_processed[categorical_cols] = self.categorical_imputer.fit_transform(df_processed[categorical_cols])
        
        logger.info("Missing values handled successfully")
        return df_processed
    
    def engineer_features(self, df):
        """Create domain-specific features for loan default prediction"""
        df_processed = df.copy()
        
        # Loan amount to income ratio
        if 'loan_amnt' in df_processed.columns and 'revenue' in df_processed.columns:
            df_processed['loan_to_income_ratio'] = df_processed['loan_amnt'] / (df_processed['revenue'] + 1)
            df_processed['loan_to_income_ratio'] = df_processed['loan_to_income_ratio'].replace([np.inf, -np.inf], 0)
        
        # FICO score categories
        if 'fico_n' in df_processed.columns:
            df_processed['fico_category'] = pd.cut(
                df_processed['fico_n'], 
                bins=[0, 580, 670, 740, 800, 850],
                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            )
        
        # Debt burden indicator
        if 'dti_n' in df_processed.columns:
            df_processed['high_debt_burden'] = (df_processed['dti_n'] > 40).astype(int)
        
        logger.info("Feature engineering completed")
        return df_processed
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_processed = df.copy()
        categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            # Handle unseen categories by fitting on current data
            df_processed[col] = df_processed[col].astype(str)
            self.label_encoders[col].fit(df_processed[col])
            df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        logger.info(f"Encoded {len(categorical_columns)} categorical variables")
        return df_processed
    
    def prepare_data(self, df, target_column='Default', test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""
        logger.info("Starting complete data preprocessing pipeline...")
        
        # Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Feature engineering
        df_processed = self.engineer_features(df_processed)
        
        # Encode categorical variables
        df_processed = self.encode_categorical(df_processed)
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column], errors='ignore')
        y = df[target_column] if target_column in df.columns else None
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        if y is not None:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Handle class imbalance with SMOTE
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_resampled)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Training set shape: {X_train_scaled.shape}")
            logger.info(f"Test set shape: {X_test_scaled.shape}")
            logger.info(f"Class distribution after SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
            
            return (X_train_scaled, X_test_scaled, y_train_resampled, y_test, self.feature_names)
        else:
            # For inference without target
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, self.feature_names
    
    def save_preprocessor(self, file_path):
        """Save preprocessor objects"""
        preprocessor_dict = {
            'numeric_imputer': self.numeric_imputer,
            'categorical_imputer': self.categorical_imputer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        joblib.dump(preprocessor_dict, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path):
        """Load preprocessor objects"""
        preprocessor_dict = joblib.load(file_path)
        self.numeric_imputer = preprocessor_dict['numeric_imputer']
        self.categorical_imputer = preprocessor_dict['categorical_imputer']
        self.scaler = preprocessor_dict['scaler']
        self.label_encoders = preprocessor_dict['label_encoders']
        self.feature_names = preprocessor_dict['feature_names']
        logger.info(f"Preprocessor loaded from {file_path}")

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = LoanDataPreprocessor()
    
    # Load sample data (replace with your actual data path)
    try:
        df = preprocessor.load_data('data/raw/LC_loans_granting_model_dataset.csv')
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df)
        
        # Save preprocessor
        preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        print("Data preprocessing completed successfully!")
        print(f"Features: {len(feature_names)}")
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")