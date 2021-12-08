from gnet import Gnet
import streamlit as st
from streamlit import caching

st.write("""# Genome network App
This app uses genomes instead of neurons to build the model!
Genome Code origin  [Genome](https://github.com/DanShai/Genome) in python by @Danshai.
""")

gn = Gnet()
st.sidebar.header('Set some Genome params')
epochs = st.sidebar.selectbox('Epochs', list(range(100,1000,100)))
mode = st.sidebar.selectbox('Mode',('Regression','Classification'))

caching.clear_cache()
gn.set_epochs(epochs)
if mode == 'Regression' :

    gn.reg()
else:
    gn.classify()
