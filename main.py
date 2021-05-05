from gnet import Gnet
import streamlit as st

st.write("""
# Genome network App
This app uses genomes instead of neurons to build the model!
Genome Code origin  [Genome](https://github.com/DanShai/Genome) in python by @Danshai.
""")

st.header(' Genome App')
gn = Gnet()

mode = st.selectbox('Mode',('Regression','Classification'))

if mode == 'Regression' :
    gn.reg()
else:
    gn.classify()
