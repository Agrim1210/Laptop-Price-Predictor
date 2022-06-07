import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe_rf.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))
st.set_page_config(page_title="Laptop Price", layout="wide")
st.title("Laptop Predictor")


# brand
company = st.selectbox('Brand', data['Company'].unique())

# type of laptop
col1, col2 = st.columns(2)
with col1:
    type = st.radio(label='Type', options=data['TypeName'].unique())
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
with col2:
# Ram
    ram = st.select_slider('How much Ram do you need?', options=[
                       2, 4, 6, 8, 12, 16, 24, 32, 64])


os = st.radio(label='OS', options=data['OpSys'].unique())
st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


# weight
weight = st.slider('Weight (KG)', 1.5, 3.0, 1.5, 0.1)
col1, col2 = st.columns(2)
with col1:
    # Touchscreen
   
    touchscreen =  st.radio(label='Touchscreen', options= ['No', 'Yes'])
with col2:
    # IPS
  
     ips=st.radio(label='IPS', options= ['No', 'Yes'])
# screen size
screen_size =st.slider(
            'Laptop Size in inches',
            min_value=14.0,
            max_value=17.5,
            value=15.5,
            step=0.5)

# resolution
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', data['Cpu Brand'].unique())
col1, col2 = st.columns(2)

with col1:
    ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

with col2:
    hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])


gpu = st.selectbox('GPU', data['Gpu Brand'].unique())


if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, os, weight,
                     touchscreen, ips, ppi, cpu, ssd, hdd, gpu])

    query = query.reshape(1, 12)
    st.title("The predicted price of this configuration is " +
             str(int(np.exp(pipe.predict(query)[0]))))
