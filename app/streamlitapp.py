# importing dependencies

import streamlit as st
import os
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import ffmpeg
import imageio


    # wide layout
st.set_page_config(layout='wide')

# sidebar
with st.sidebar:
    st.image('https://img.freepik.com/free-vector/big-data-circular-colorful-visualization-visual-data-complexity_1217-1858.jpg?w=740&t=st=1692160237~exp=1692160837~hmac=491e7851ece40840aab66e00073c2bd6d09e2d7264640ba17abea0fd78bb8e12')
    st.title('LipBuddy')
    
    st.info('URL to the Model - https://www.kaggle.com/code/prolevelnoob/shubh-lipreader')
    
    

st.title('LipBuddy an implementation of the LipNet Model')
# generating list of options or videos

options=os.listdir(os.path.join('..','data','s1'))
print(options)
selected_video=st.selectbox('Choose Video',options)

# generating columns
col1, col2=st.columns(2)

if options:
    # rendering the video
    with col1:
        st.info('Displaying the selected test video in MP4 format')
        file_path=os.path.join('..','data','s1',selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
    
        # rendering in app
        video=open('test_video.mp4','rb')
        video_bytes=video.read()
        st.video(video_bytes)
    
    with col2:
        st.info('This is all the ML Model sees when making the prediction - represented as a gif')
        video,annotations=load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('anime.gif',video,fps=10)
        st.image('anime.gif',width=400)
        
        st.info('This is output of the ML Model as tokens')
        model=load_model()
        yhat = model.predict(tf.expand_dims(video,axis=0))
        decoder=tf.keras.backend.ctc_decode(yhat,[75],greedy=True)[0][0].numpy()
        st.text(decoder)
        
        # covert the prediction to char and to strings
        st.info('displaying the Decoded raw numbers as words')
        converted_prediction=tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
    
    
st.text('developed by : Shubh Rai')
st.markdown("""
- This model is an implementation of   [LipNet: End-to-End Sentence-level Lipreading](https://arxiv.org/abs/1611.01599) .
- Associated Code for Paper: https://github.com/rizkiarm/LipNet
- TensorFlow dataPipeline reference -https://www.tensorflow.org/api_docs/python/tf/data
- CTC_LOSS referance - https://keras.io/examples/audio/ctc_asr/#model
""")