
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestRegressor
import re

# Set page configuration
st.set_page_config(page_title="ATLL Genetic Mutation Survival Prediction Tool", layout="wide")

# Title and introduction
st.title("ATLL Genetic Mutation Survival Prediction Tool")
st.markdown("Predict survival outcomes based on patient age and genetic mutations")

# Data preparation - same as your notebook
data = {
    'ATL_no': list(range(1, 31)),
    'Sex': ['Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male',
            'Female', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male',
            'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female'],
    'Age': [83, 59, 28, 66, 74, 53, 48, 64, 63, 36,
            70, 54, 37, 41, 51, 57, 36, 67, 65, 74,
            37, 60, 54, 59, 64, 61, 40, 40, 63, 70],
    'Subtype': ['A', 'L', 'A', 'L', 'L', 'A', 'L', 'L', 'L', 'L',
                'L', 'L', 'A', 'L', 'L', 'C', 'A', 'A', 'A', 'A',
                'C/S', 'C/S', 'A', 'L', 'A', 'S', 'L', 'A', 'A', 'A'],
    'OS_days': [33, 382, 543, 363, 160, 88, 225, 215, 739, 52,
                451, 23, 17, 771, 312, 2554, 75, 175, 392, 1191,
                6345, 126, 503, 176, 73, 102, 351, 322, 192, 153],
    'Mutations_text': [
        'KIT, APC, MYC, TP53, DDX3X, TSC1',
        'NRAS, POT1, CEBPA, KEAP1, AR, NOTCH1, MCL1',
        'ALK, BCL6, RIPK1, GATA3, IGF1R, TP53, CEBPA',
        'NOTCH2, DDR2, GATA3',
        'KDR, FAT1, FGFR2, AKT1, TP53, ERBB2, MED12',
        'XPO1, TBL1XR1, PDGFRA, KIT, FGFR4, EP300, RICTOR',
        'KDR, CARD11, NKX2-1 (2)',
        'TNFAIP3, TSC1, BRCA2, NOTCH1',
        'CARD11, NOTCH1, KRAS, POT1',
        'CEBPA, RHOA, PBRM1, CD79A, STAG2',
        'NRAS, VHL, APC, EGFR, PTCH1, ASXL1, KMT2A',
        'FAT1, HIST1H1E, PLCG2',
        'FGFR4, APC, FAT1, SPEN',
        'EZH2, DDX3X, FAT1, PTCH1',
        'TP53, EP300, PALB2, PTCH1, TRAF3',
        'APC, PDGFRB, TET2, DNMT3A',
        'KDR, NOTCH1, TP53, SPOP',
        'AKT2, CDKN2A, EP300, FAT1, FBXW7, FLT3, PBRM1, SPEN',
        'SETBP1, EP300',
        'FGFR3, FAT1',
        'CDH1, EP300',
        'AKT2, ALK, CD79A, EP300, ZMYM3, AR',
        'DDR2, IDH1, TBL1XR1, TP53, ZMYM3',
        'FGFR3, FLT1, KRAS, P2RY8, TET2',
        'AR, JAK3, PDGFRA, SETBP1, TP53, XPO1, ZFHX4',
        'AKT1, AXL, PALB2',
        'BCL6, MDM4',
        'CDH1, HIST1H1E, NOTCH1',
        'NOTCH1, APC, APC, GATA2, KLF2, NTRK1, SMARCB1, SPEN, TBL1XR1, TCF3',
        'ALK, ALK, CDKN2A, ERBB3, FAS, FAT1, HRAS, KLF2, PIK3CD, PIK3CD, TBL1XR1'
    ],
    'Mutations_count': [5, 5, 7, 3, 7, 6, 4, 3, 3, 5, 
                        6, 3, 3, 4, 5, 3, 3, 7, 2, 2, 
                        2, 6, 5, 5, 7, 2, 2, 3, 10, 11]
}

# Create DataFrame
df = pd.DataFrame(data)

# Extract all unique mutations
all_mutations = set()
for mutations_text in df['Mutations_text']:
    genes = re.findall(r'[A-Z0-9]+', mutations_text)
    all_mutations.update(genes)
all_mutations = sorted([m for m in all_mutations if not m.isdigit()])

# Create mutation features
for mutation in all_mutations:
    df[f'has_{mutation}'] = df['Mutations_text'].apply(lambda x: 1 if mutation in x else 0)

# Prepare features for model
X = df.filter(regex='^has_|Mutations_count|Age').copy()
y = df['OS_days']

# Train Random Forest model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Sidebar with patient info
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 20, 90, 60)

# Create mutation selection with search
st.sidebar.header("Genetic Mutations")
mutation_search = st.sidebar.text_input("Search mutations")

# Filter mutations based on search
filtered_mutations = [m for m in all_mutations if mutation_search.upper() in m]

# Create checkboxes for mutations (max height with scrolling)
st.sidebar.write("Select mutations:")
mutation_container = st.sidebar.container()
selected_mutations = []

# Use columns to display mutations in a compact way
all_selected = st.sidebar.checkbox("Select All")
if all_selected:
    selected_mutations = filtered_mutations
else:
    # Create a scrollable area for mutations
    with mutation_container:
        for mutation in filtered_mutations:
            if st.checkbox(mutation, key=mutation):
                selected_mutations.append(mutation)

# Prediction button
predict_button = st.sidebar.button("Predict Survival")

# Main area for results
if predict_button:
    # Count mutations
    mutation_count = len(selected_mutations)
    
    # Create feature vector
    features = pd.DataFrame({
        'Age': [age],
        'Mutations_count': [mutation_count]
    })
    
    # Add mutation features
    for mutation in all_mutations:
        features[f'has_{mutation}'] = 1 if mutation in selected_mutations else 0
    
    # Ensure all features exist and reorder
    for col in X.columns:
        if col not in features.columns:
            features[col] = 0
    features = features[X.columns]
    
    # Make prediction
    predicted_days = rf_model.predict(features)[0]
    
    # Find similar patients
    similar_patients = df[(df['Mutations_count'] >= mutation_count - 1) & 
                          (df['Mutations_count'] <= mutation_count + 1) &
                          (df['Age'] >= age - 10) & 
                          (df['Age'] <= age + 10)]
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Patient Profile")
        st.write(f"**Age:** {age}")
        st.write(f"**Number of Mutations:** {mutation_count}")
        
        if selected_mutations:
            st.write("**Selected Mutations:**")
            st.write(", ".join(selected_mutations))
        else:
            st.write("**No mutations selected**")
    
    with col2:
        st.header("Survival Prediction")
        st.markdown(f"### Predicted Survival: **{predicted_days:.0f} days**")
        
        if len(similar_patients) > 0:
            st.write(f"Based on {len(similar_patients)} similar patients:")
            st.write(f"- Average survival: {similar_patients['OS_days'].mean():.0f} days")
            st.write(f"- Median survival: {similar_patients['OS_days'].median():.0f} days")
    
    # Show survival curves by subtype
    st.header("Reference Survival Curves")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for subtype in df['Subtype'].unique():
        kmf = KaplanMeierFitter()
        mask = df['Subtype'] == subtype
        if mask.sum() > 0:
            kmf.fit(df[mask]['OS_days'], label=f'Subtype {subtype}')
            kmf.plot(ax=ax)
    
    plt.title('Survival Curves by ATLL Subtype')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)
    
    # Important note
    st.markdown("---")
    st.warning("""
    **Important Note:** This model is based on a small dataset (30 patients) and should be considered 
    exploratory only. It is not intended for clinical decision-making. The predictions represent 
    statistical associations and not causal relationships.
    """)
else:
    # Display information when first loading
    st.info("""
    ### Instructions
    1. Set the patient's age using the slider in the sidebar
    2. Select genetic mutations present in the patient's profile
    3. Click "Predict Survival" to generate predictions
    
    This tool uses a machine learning model trained on 30 ATLL patients to predict survival 
    based on age and genetic mutation profile.
    """)
    
    # Display overview of data
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients", len(df))
    
    with col2:
        st.metric("Unique Mutations", len(all_mutations))
    
    with col3:
        st.metric("Median Survival", f"{df['OS_days'].median():.0f} days")
    
    # Display subtype breakdown
    st.subheader("ATLL Subtypes")
    subtype_counts = df['Subtype'].value_counts().reset_index()
    subtype_counts.columns = ['Subtype', 'Count']
    
    st.bar_chart(subtype_counts.set_index('Subtype'))
