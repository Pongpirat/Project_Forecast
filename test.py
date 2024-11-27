import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import plotly
import statsmodels
import sklearn
import pmdarima

# Print versions
st.write(f"Streamlit version: {st.__version__}")
st.write(f"Numpy version: {np.__version__}")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"Matplotlib version: {matplotlib.__version__}")
st.write(f"Plotly version: {plotly.__version__}")
st.write(f"Statsmodels version: {statsmodels.__version__}")
st.write(f"Scikit-learn version: {sklearn.__version__}")
st.write(f"pmdarima version: {pmdarima.__version__}")
