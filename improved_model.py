# improved_model.py - FIXED VERSION WITH CLASS IMBALANCE HANDLING
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

print("🚀 IMPROVED MODEL - Handling Class Imbalance")
print("=" * 60)

# 1. LOAD DATA
print("📊 Loading data...")
df = pd.read_csv('data/raw/LC_loans_granting_model_dataset.csv', low_memory=False)
print(f"✅ Data loaded! Shape: {df.shape}")

# 2. PREPROCESSING
print("\n🔄 Advanced preprocessing...")

# Select numeric features only
numeric_features = ['revenue', 'dti_n', 'loan_amnt', 'fico_n', 'experience_c']
X = df[numeric_features].fillna(0)
y = df['Default']

print(f"Features: {X.columns.tolist()}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 3. HANDLE CLASS IMBALANCE WITH SMOTE
print("\n⚖️ Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"After SMOTE - Features: {X_resampled.shape}, Target: {y_resampled.value_counts().to_dict()}")

# 4. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 5. TRAIN BETTER MODEL
print("\n🤖 Training Improved Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# 6. EVALUATE
print("\n📊 Evaluating improved model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"🎯 ACCURACY: {accuracy:.4f}")
print(f"📈 AUC SCORE: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. FEATURE IMPORTANCE
print("\n🔝 Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# 8. SAVE IMPROVED MODEL
print("\n💾 Saving improved model...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/improved_model.pkl')

# Also save the feature names for the app
feature_info = {
    'features': numeric_features,
    'feature_importance': feature_importance.to_dict()
}
joblib.dump(feature_info, 'models/feature_info.pkl')

print("✅ Improved model saved as 'models/improved_model.pkl'")

print("\n🎉 IMPROVED MODEL READY! Now run the apps.")