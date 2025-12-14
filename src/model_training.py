import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import joblib
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanDefaultModel:
    """Complete loan default prediction model with multiple algorithms"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=42, n_jobs=-1),
            'lightgbm': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
            'catboost': CatBoostClassifier(random_state=42, verbose=False),
            'logistic_regression': LogisticRegression(random_state=42, n_jobs=-1)
        }
        self.trained_models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'precision': clf_report['1']['precision'],
                'recall': clf_report['1']['recall'],
                'f1_score': clf_report['1']['f1-score'],
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'confusion_matrix': cm,
                'classification_report': clf_report
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return None
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='xgboost', n_trials=50):
        """Hyperparameter tuning using Optuna"""
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
                
            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
                }
                model = CatBoostClassifier(**params, random_state=42, verbose=False)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best trial for {model_name}: Score = {study.best_trial.value:.4f}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        return study.best_params
    
    def train_models(self, X_train, X_test, y_train, y_test, tune_hyperparams=False):
        """Train multiple models and evaluate performance"""
        self.model_results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                if tune_hyperparams:
                    logger.info(f"Tuning hyperparameters for {name}...")
                    best_params = self.hyperparameter_tuning(X_train, y_train, name, n_trials=30)
                    
                    # Update model with best parameters
                    if name == 'xgboost':
                        model = xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1)
                    elif name == 'lightgbm':
                        model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
                    elif name == 'catboost':
                        model = CatBoostClassifier(**best_params, random_state=42, verbose=False)
                    elif name == 'random_forest':
                        model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
                
                # Train model
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                
                # Evaluate model
                results = self.evaluate_model(model, X_test, y_test, name)
                if results:
                    self.model_results[name] = results
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Identify best model based on AUC score
        if self.model_results:
            self.best_model_name = max(self.model_results, key=lambda x: self.model_results[x]['auc_score'])
            self.best_model = self.trained_models[self.best_model_name]
            logger.info(f"Best model: {self.best_model_name} with AUC: {self.model_results[self.best_model_name]['auc_score']:.4f}")
        
        return self.model_results
    
    def explain_model(self, model, X_test, feature_names, model_name):
        """Generate SHAP explanations for model predictions"""
        try:
            logger.info(f"Generating SHAP explanations for {model_name}...")
            
            # Create SHAP explainer
            if model_name in ['xgboost', 'lightgbm', 'random_forest']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_test)
            
            # Calculate SHAP values
            shap_values = explainer(X_test)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'models/shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap_df = pd.DataFrame({
                'features': feature_names,
                'importance': np.abs(shap_values.values).mean(0)
            }).sort_values('importance', ascending=True)
            
            plt.barh(shap_df['features'][-20:], shap_df['importance'][-20:])
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Mean |SHAP value|')
            plt.tight_layout()
            plt.savefig(f'models/feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation for {model_name}: {e}")
            return None
    
    def save_models(self, path='models/'):
        """Save trained models and artifacts"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.trained_models.items():
            joblib.dump(model, f'{path}{name}_model.pkl')
        
        # Save results
        results_df = pd.DataFrame({
            'model': list(self.model_results.keys()),
            'accuracy': [self.model_results[m]['accuracy'] for m in self.model_results],
            'auc_score': [self.model_results[m]['auc_score'] for m in self.model_results],
            'precision': [self.model_results[m]['precision'] for m in self.model_results],
            'recall': [self.model_results[m]['recall'] for m in self.model_results],
            'f1_score': [self.model_results[m]['f1_score'] for m in self.model_results]
        })
        
        results_df.to_csv(f'{path}model_results.csv', index=False)
        
        # Save best model separately
        if self.best_model:
            joblib.dump(self.best_model, f'{path}best_model.pkl')
            logger.info(f"Best model ({self.best_model_name}) saved to {path}best_model.pkl")
        
        logger.info("All models and results saved successfully")

def run_complete_training():
    """Complete training pipeline"""
    from data_preprocessing import LoanDataPreprocessor
    
    # Initialize preprocessor
    preprocessor = LoanDataPreprocessor()
    
    try:
        # Load and preprocess data
        df = preprocessor.load_data('data/raw/LC_loans_granting_model_dataset.csv')
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df)
        
        # Initialize and train models
        model_trainer = LoanDefaultModel()
        results = model_trainer.train_models(X_train, X_test, y_train, y_test, tune_hyperparams=True)
        
        # Generate explanations for best model
        if model_trainer.best_model:
            shap_values = model_trainer.explain_model(
                model_trainer.best_model, 
                X_test, 
                feature_names, 
                model_trainer.best_model_name
            )
        
        # Save everything
        model_trainer.save_models()
        preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        # Print final results
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        for model_name, result in results.items():
            print(f"{model_name.upper():<20} AUC: {result['auc_score']:.4f} | Accuracy: {result['accuracy']:.4f}")
        
        print(f"\nBest Model: {model_trainer.best_model_name}")
        print(f"Best AUC Score: {results[model_trainer.best_model_name]['auc_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in complete training pipeline: {e}")
        raise

if __name__ == "__main__":
    run_complete_training()