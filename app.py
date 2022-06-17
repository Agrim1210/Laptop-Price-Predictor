import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe_rf.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))
st.set_page_config(page_title="Laptop Price", layout="wide")
st.title("Laptop Hub")


# brand
company = st.selectbox('Brand', data['Company'].unique())

# type of laptop
type_col, ram_col,os_col = st.columns([3,3,2])
with type_col:
    type = st.radio(label='Type', options=data['TypeName'].unique())
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
with ram_col:
# Ram
    ram = st.select_slider('How much Ram do you need?', options=[
                       2, 4, 6, 8, 12, 16, 24, 32, 64])

with os_col:
    os = st.radio(label='OS', options=data['OpSys'].unique())
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

weight_col,touch_col,ips_col=st.columns([4,1,1])
with weight_col:
# weight
    weight = st.slider('Weight (KG)', 1.5, 3.0, 1.5, 0.1)

with touch_col:
    # Touchscreen
   
    touchscreen =  st.radio(label='Touchscreen', options= ['No', 'Yes'])
with ips_col:
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

cpu_col, ssd_col,hdd_col,gpu_col = st.columns(4)
with cpu_col:
    # cpu
    cpu = st.selectbox('CPU', data['Cpu Brand'].unique())

with ssd_col:
    ssd = st.select_slider('SSD(in GB)',options=[ 0, 8, 128, 256, 512, 1024])

with hdd_col:
    hdd = st.select_slider('HDD(in GB)',options=[ 0, 8, 128, 256, 512, 1024])

with gpu_col:
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
    st.title("The predicted price of this configuration is ₹" +
             str(int(np.exp(pipe.predict(query)[0]))))
