# 🏦 Loan Default Prediction Web App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive machine learning web application for predicting loan default risk using real-world lending data. Built for fintech applications, hackathons, and portfolio showcases.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This application predicts the likelihood of loan default based on applicant information using machine learning models trained on real lending data. The system provides:

- **Real-time predictions** with probability scores
- **Model interpretability** using SHAP values
- **Interactive dashboard** for loan officers
- **Professional deployment** ready for production

## ✨ Features

### 🤖 Machine Learning
- Multiple ensemble models (Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning with Optuna
- SHAP explanations for model interpretability
- Comprehensive evaluation metrics

### 🎨 User Interface
- Interactive Streamlit dashboard
- Real-time prediction visualization
- Feature importance charts
- User-friendly input forms

### 🚀 Production Ready
- Modular code architecture
- Error handling and logging
- Docker containerization
- FastAPI REST endpoints
- Model monitoring with drift detection

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python, FastAPI |
| **ML Framework** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly, SHAP |
| **Deployment** | Docker, Heroku/AWS/GCP |
| **Version Control** | Git, GitHub |

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # On Mac/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   *If requirements.txt doesn't exist, install manually:*
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn xgboost lightgbm catboost shap
   ```

4. **Download/Prepare the dataset**
   ```bash
   # Place your dataset in the data/ folder
   # The app expects LendingClub data (2018-2020) by default
   ```

5. **Train the model (optional)**
   ```bash
   # If you have training scripts
   python src/train_model.py
   ```

## 🚀 Usage

### Running the Web Application

1. **Navigate to the app directory**
   ```bash
   cd loan-default-prediction
   ```

2. **Activate the virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Launch the Streamlit app**
   ```bash
   streamlit run app/frontend.py
   ```

4. **Access the application**
   - Open your web browser
   - Go to `http://localhost:8501`
   - The application should now be running!

### Using the Application

1. **Input Applicant Information**
   - Fill in the form with applicant details (income, credit score, loan amount, etc.)
   - Use the sidebar sliders and input fields

2. **Get Predictions**
   - Click the "Predict" button
   - View the default probability and risk category

3. **Interpret Results**
   - See SHAP force plots explaining the prediction
   - View feature importance charts
   - Check confidence metrics

## 📁 Project Structure

```
loan-default-prediction/
├── app/
│   ├── frontend.py              # Streamlit web application
│   └── fastapi_app.py           # REST API backend
├── models/
│   ├── improved_model.pkl       # Trained ML model
│   └── feature_info.pkl         # Feature metadata
├── src/
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── train_model.py           # Model training pipeline
│   ├── predict.py               # Prediction functions
│   └── utils.py                 # Utility functions
├── notebooks/
│   └── eda_modeling.ipynb       # Exploratory data analysis
├── tests/
│   ├── test_preprocessing.py    # Unit tests
│   └── test_predictions.py
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Cleaned data
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🤖 Model Details

### Algorithms Used
1. **Random Forest** - Baseline ensemble model
2. **XGBoost** - Gradient boosting with regularization
3. **LightGBM** - Efficient gradient boosting framework
4. **CatBoost** - Handles categorical features natively

### Performance Metrics
- **Accuracy**: Measures overall correctness
- **Precision**: Proportion of true defaults among predicted defaults
- **Recall**: Proportion of actual defaults correctly identified
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Model discrimination ability

### Feature Engineering
- Missing value imputation
- Feature scaling and normalization
- Categorical encoding
- Interaction feature creation
- Handling class imbalance (SMOTE)

## 🌐 Deployment

### Local Deployment with Docker

1. **Build the Docker image**
   ```bash
   docker build -t loan-default-app .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 loan-default-app
   ```

### Cloud Deployment Options

**Heroku:**
```bash
heroku create loan-default-predictor
git push heroku main
```

**AWS Elastic Beanstalk:**
```bash
eb init -p python-3.8 loan-default-app
eb create loan-default-env
```

**Google Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/your-project/loan-default-app
gcloud run deploy --image gcr.io/your-project/loan-default-app
```

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Run `pip install missing-module-name` |
| **Streamlit not recognized** | Use `python -m streamlit run app/frontend.py` |
| **Port already in use** | Change port: `streamlit run app/frontend.py --server.port 8502` |
| **Model loading error** | Check file paths and pickle version compatibility |
| **Memory issues** | Reduce dataset size or use chunk processing |

### Debugging Tips
1. Check virtual environment is activated
2. Verify all dependencies are installed
3. Check file paths in your code
4. Look at error logs for specific messages
5. Test components individually

## 👥 Contributing

We welcome contributions! Here's how to help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Write clear commit messages
- Add tests for new functionality
- Update documentation as needed
- Follow PEP 8 style guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LendingClub for providing the dataset
- Streamlit team for the amazing framework
- Open-source ML community for libraries and tools

## 📞 Support

For questions, issues, or contributions:
- Open an [Issue](https://github.com/JinxWycman/loan-default-prediction/issues)
- Check the [Discussions](https://github.com/JInxWycman/loan-default-prediction/discussions)
- Email: machariajoseph1422@gmail.com

---

**⭐ If you find this project useful, please give it a star on GitHub!**

