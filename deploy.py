import sklearn
import streamlit as st
import pickle
import pandas as pd

model=pickle.load(open('log_model.pkl','rb'))
tfi=pickle.load(open('tfidf.pkl','rb'))
labe=pickle.load(open('labeling.pkl','rb'))
lab_mo=pickle.load(open('lab_model.pkl','rb'))

st.set_page_config(layout="wide")
st.title('Sentiment Analysis')
col_l,col_r=st.columns([3,1])

with col_l:
  input=st.text_area('Enter your Text here......',height=500)

label={0:'Negative',1:'Neutral',2:'Positive'}

with col_r:
  if st.button('Predict Sentiment'):
    vector=tfi.transform([input])
    pred=model.predict(vector)[0]
    pred_prob=model.predict_proba(vector)
    c_pred=lab_mo.predict(vector)[0]
    c_label=labe.inverse_transform([c_pred])[0]
    sent_label=label[pred]
    neg,neu,pos=pred_prob[0]

    if pred==2:
      st.write(f'Positive Sentiment')
      st.write(f'Most likely for {c_label}')
    if pred==1:
      st.write(f'Neutral Sentiment')
      st.write(f'Most likely for {c_label}')
    if pred==0:
      st.write(f'Negative Sentiment')
      st.write(f'Most likely for {c_label}')

    tab={'Sentiment':['Positive','Neutral','Negative'],'Probability':[round(pos*100,2),round(neu*100,2),round(neg*100,2)]}
    st.table(pd.DataFrame(tab))
