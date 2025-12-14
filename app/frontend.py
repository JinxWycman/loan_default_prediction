# app/frontend.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import requests
from PIL import Image
import io


st.set_page_config(page_title="Smart Loan ", page_icon="🏦", layout="wide")

# Custom CSS for background image
def set_background_image(image_url):
    """Set background image from URL"""
    try:
        # Download image from URL
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Apply background CSS
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{img_str}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* Make content more readable on background */
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.95);
                padding: 2rem;
                border-radius: 10px;
                backdrop-filter: blur(5px);
            }}
            
            /* Style headers for better visibility */
            h1, h2, h3 {{
                color: #1f3a60;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        return True
    except Exception as e:
        st.sidebar.error(f"❌ Couldn't load background image: {e}")
        return False

# Alternative method using direct URL (simpler)
def set_background_direct(image_url):
    """Set background using direct URL (no download)"""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }}
        
        h1, h2, h3 {{
            color: #1f3a60;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("🏦 Smart Loan prediction ")
st.markdown("Predict loan defaults effectively.")


# Currency conversion rates (approximate)
USD_TO_KSH = 130  # 1 USD = ~130 KSH

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/improved_model.pkl')
        feature_info = joblib.load('models/feature_info.pkl')
        return model, feature_info
    except:
        st.error("❌ Model not found. Please train the model first.")
        return None, None

model, feature_info = load_model()

# Sidebar for settings
st.sidebar.header("ℹ️ App Info")


# ... your existing sidebar code above ...

st.sidebar.markdown("---")
st.sidebar.header("👨‍💻 Developer Info")

st.sidebar.markdown("""
**Developed by:** Joseph Thuo  
**Role:** Senior Data Scientist  
**Specialization:** Machine Learning & web development

**Contact & Portfolio:**
""")

st.sidebar.markdown("""
[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-Visit-blue)](http://my-portfolio-3wrw.onrender.com)
[![LinkedIn](https://img.shields.io/badge/💼_LinkedIn-Connect-blue)](https://www.linkedin.com/in/machariajosepht/)
[![GitHub](https://img.shields.io/badge/🐙_GitHub-Follow-black)](https://github.com/JinxWycman)
[![Email](https://img.shields.io/badge/📧_Email-Contact-red)](mailto:machariajoseph1422@gmail.com)
""")

# ... your existing sidebar code below continues ...
set_background_direct("https://www.shutterstock.com/image-photo/dark-blue-degradation-view-deep-600nw-2557945519.jpg")

# Currency settings
currency = st.sidebar.radio("Select Currency", ["Kenyan Shillings (KSH)", "US Dollars (USD)"])
    
if currency == "Kenyan Shillings (KSH)":
    exchange_rate = USD_TO_KSH
    st.sidebar.info(f"Exchange Rate: 1 USD = {exchange_rate} KSH")
else:
    exchange_rate = 1
    st.sidebar.info("Using US Dollars")

# Kenyan-specific loan information
st.sidebar.header("🇰🇪 Kenya Loan Info")
st.sidebar.info("""
**Typical Kenyan Ranges:**
- **Minimum Wage**: ~15,000 KSH/month
- **Average Income**: 300,000-800,000 KSH/year
- **Small Business Loans**: 50K - 2M KSH
- **Personal Loans**: 20K - 1M KSH
- **Mortgages**: 1M - 10M+ KSH
""")

if model:
    st.header("👤 Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if currency == "Kenyan Shillings (KSH)":
            # Kenyan-specific income ranges starting from 100,000 KSH
            annual_income_ksh = st.number_input(
                "Annual Income (KSH)", 
                min_value=100000, 
                max_value=20000000, 
                value=800000, 
                step=50000,
                help="Typical range: 100,000 - 5,000,000 KSH per year"
            )
            annual_income_usd = annual_income_ksh / exchange_rate
            st.write(f"💵 Equivalent: ${annual_income_usd:,.0f} USD")
            
            # Kenyan loan amount ranges
            loan_amount_ksh = st.number_input(
                "Loan Amount (KSH)", 
                min_value=50000, 
                max_value=10000000, 
                value=500000, 
                step=50000,
                help="Typical range: 50,000 - 5,000,000 KSH"
            )
            loan_amount_usd = loan_amount_ksh / exchange_rate
            st.write(f"💵 Equivalent: ${loan_amount_usd:,.0f} USD")
            
            # Show monthly equivalents
            monthly_income_ksh = annual_income_ksh / 12
            st.caption(f"📅 Monthly Income: KSh {monthly_income_ksh:,.0f}")
            
        else:
            annual_income_usd = st.number_input("Annual Income (USD)", min_value=1000, max_value=500000, value=20000, step=1000)
            loan_amount_usd = st.number_input("Loan Amount (USD)", min_value=500, max_value=50000, value=5000, step=500)
    
    with col2:
        credit_score = st.slider(
            "Credit Score", 
            300, 850, 650,
            help="Kenyan credit scores typically range 300-850"
        )
        
        dti_ratio = st.slider(
            "Debt-to-Income Ratio (%)", 
            0.0, 100.0, 25.0, 
            step=0.1,
            help="Monthly debt payments ÷ Monthly income"
        )
        
        experience = st.slider(
            "Years of Work Experience", 
            0, 40, 3,
            help="Total years in employment"
        )
        
        # Employment type (Kenyan context)
        employment_type = st.selectbox(
            "Employment Type",
            ["Formal Employment", "Informal Employment", "Self-Employed", "Business Owner"]
        )
    
    # Additional Kenyan context
    st.subheader("🇰🇪 Kenyan Market Context")
    context_col1, context_col2, context_col3 = st.columns(3)
    
    with context_col1:
        st.metric("Avg. Kenyan Salary", "KSh 400,000/yr")
        
    with context_col2:
        st.metric("Typical Loan Term", "12-36 months")
        
    with context_col3:
        st.metric("Interest Rates", "12-18% p.a.")
    
    # Predict button
    if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):
        # Create feature array in correct order (using USD values for model)
        features = np.array([[
            annual_income_usd,    # revenue (converted to USD)
            dti_ratio,            # dti_n  
            loan_amount_usd,      # loan_amnt (converted to USD)
            credit_score,         # fico_n
            experience            # experience_c
        ]])
        
        try:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            st.header("🎯 Prediction Results")
            
            # Display results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 0:
                    st.success("""
                    ✅ **LOW RISK** 
                    
                    **Recommendation:** APPROVE loan
                    """)
                else:
                    st.error("""
                    🚨 **HIGH RISK**
                    
                    **Recommendation:** REJECT or further review needed
                    """)
            
            with result_col2:
                # Probability gauge
                st.metric("Default Probability", f"{probability:.1%}")
                
                # Risk level with Kenyan context
                if probability < 0.2:
                    st.success("🟢 LOW RISK - Good candidate")
                    st.progress(probability)
                elif probability < 0.4:
                    st.warning("🟡 MEDIUM RISK - Review carefully") 
                    st.progress(probability)
                else:
                    st.error("🔴 HIGH RISK - Not recommended")
                    st.progress(probability)
            
            # Display loan details in both currencies
            st.header("💰 Loan Details")
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.subheader("Kenyan Shillings (KSH)")
                st.write(f"**Annual Income:** KSh {annual_income_ksh:,.0f}")
                st.write(f"**Loan Amount:** KSh {loan_amount_ksh:,.0f}")
                st.write(f"**Loan-to-Income Ratio:** {(loan_amount_ksh/annual_income_ksh)*100:.1f}%")
                st.write(f"**Employment:** {employment_type}")
                
            with detail_col2:
                st.subheader("US Dollars (USD)")
                st.write(f"**Annual Income:** ${annual_income_usd:,.0f}")
                st.write(f"**Loan Amount:** ${loan_amount_usd:,.0f}")
                st.write(f"**Loan-to-Income Ratio:** {(loan_amount_usd/annual_income_usd)*100:.1f}%")
                st.write(f"**Credit Score:** {credit_score}")
            
            # Kenyan banking recommendations
            st.header("🏦 Kenyan Banking Guidelines")
            
            if (loan_amount_ksh/annual_income_ksh) > 0.5:
                st.warning("⚠️ **High Loan-to-Income Ratio**: Consider reducing loan amount")
            elif dti_ratio > 40:
                st.warning("⚠️ **High Debt Burden**: Applicant may struggle with repayments")
            else:
                st.success("✅ **Within Conservative Limits**")
                    
            # Feature importance explanation
            st.header("📈 Key Decision Factors")
            st.write("Most important features in prediction:")
            feature_importance = {
                'Credit Score': 'High impact on repayment ability',
                'Debt-to-Income': 'Current financial obligations',
                'Loan Amount': 'Size of requested loan',
                'Annual Income': 'Repayment capacity',
                'Experience': 'Employment stability'
            }
            
            for feature, explanation in feature_importance.items():
                st.write(f"• **{feature}**: {explanation}")
                    
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.warning("Please run the improved model first: `python improved_model.py`")

# Footer with Kenyan context
st.markdown("---")
st.caption("""
**🇰🇪 Kenyan Loan Default Prediction System**  
*Developed by Joseph Thuo Macharia*
""")