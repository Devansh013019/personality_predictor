import streamlit as st
import pandas as pd
import joblib


model=joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

dict={
    'Yes':1,
    'No':0,
}

st.title('Personality predictor')

st.markdown("ENTER THE REQUIRED FIELD")

time=st.number_input('Time Spent Alone',0,24,4)
stage=st.selectbox('Stage Fear',['Yes','No'])
event=st.slider('Social Event Attendance',0,10,4)
out=st.number_input('Going Outside')
drain=st.selectbox('Drain After Socialising',['Yes','No'])
frnd=st.number_input('Frind Circle Size',0,20,5)
post=st.slider('Post Frequency',0,10,4)


if st.button('Predict'):
    dat={
        'Time_spent_Alone':[time],
        'Stage_fear':[dict[stage]],
        'Social_event_attendance':[event],
        'Going_outside':[out],
        'Drained_after_socializing':[dict[drain]],
        'Friends_circle_size':[frnd],
        'Post_frequency':[post]
    }

    input_df=pd.DataFrame(dat)
    input_scaled=scaler.transform(input_df)
    prediction=model.predict(input_scaled)
    prediction=prediction[0]
    if prediction==0:
        st.success('You are Extrovert')
    else:
        st.success('You are Introvert')