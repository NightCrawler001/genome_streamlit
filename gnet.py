
import warnings
import time
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from GN.NN.NClassifier import NClassifier
from GN.NN.NRegressor import NRegressor
from MTSNE.mtsne import Mtsne
from Threads.PThread import PThread
from Threads.SetInterval import SetInterval
import streamlit as st
# import plotly.graph_objects as go
warnings.filterwarnings("ignore", category=RuntimeWarning)
import altair as alt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler

class Gnet:
    def __init__(self):

        self.scaler = StandardScaler() # MinMaxScaler(feature_range=(-1, 1))
        self.net = None
        self.datas = None
        self.interval = None
        self.the_plot = None
        self.prev_score = None
        self.epochs = 200


    def set_epochs(self,epochs=200):
        self.epochs = epochs

    def reg(self):

        # nrOpts = {  "opx": 2, "depth": 2, "nvars": None, "pvc": .8, "pf": 1., 
        #             "cross": .6, "mut": .4, "mrand": 1.}
        # gOpts = {   "mxepoch": 900, "bsize": 500, "bupdate": 50, "fraction": .8,
        #             "history": 5, "mxtries": 15, "mode": 'REG'}                


        nrOpts = {  "opx": 2, "depth": 1, "nvars": None, "pvc": .8, "pf": 1., 
                    "cross": .6, "mut": .2, "mrand": 1.}
        gOpts = {   "mxepoch": self.epochs, "bsize": 500, "bupdate": 50, "fraction": .8,
                    "history": 5, "mxtries": 15, "mode": 'REG'}                

        # maxepoch >= 1000  --> good

        print(nrOpts)
        print(gOpts)

        X0 = np.linspace(0., 10., num=500)
        Y0 = (X0 - 2) * np.cos(2 * X0)

        Y = Y0[:-5]
        nY = Y0[-5:]
        X = X0[:-5]
        X = X[:, None]
        nX = X0[-5:]
        nX = nX[:, None]
        self.datas = {"X": X, "Y": Y, "nX": nX, "nY": nY}
        inp = X.shape[1]
        l_dims = [inp, 1]
        
        print(X.shape,Y.shape)
        

        prg_bar = st.empty()
        st.subheader(' Real: ')
        st.latex(r''' cos(2*x0) * (x0-2)  ''')
        
        self.the_plot = st.empty() 
        self.prev_score = np.inf

        self.net = NRegressor(dim=l_dims, datas=self.datas, nr_opts=nrOpts, g_opts=gOpts)
        self.net.setLoading(True)
        lf = PThread(target=self.net.train)
        st.report_thread.add_report_ctx(lf)
        lf.start()


        while(True):
            self.reg_thread()
            e,m = self.net.get_cur_epoc()-1 , self.net.get_max_epoc()
            prg_bar.text(f' peogress: {np.round(100*e/m,2)}% ')
            on = self.net.getLoading()
            if not on:
                break
            time.sleep(2.)

        st.subheader(' Predicted')
        fnc = self.net.get_best_net()[0]
        st.latex(f'{str(fnc).strip()}')



    def plot_reg(self, X, Y, yh):
        bs = self.net.get_best_score()
        if (  bs < self.prev_score  ):
            source = pd.DataFrame({'x':X.ravel() , 'y': Y , 'z': yh },columns=['x','y','z'],index=X.ravel())
            real = alt.Chart(source).mark_line(opacity=1).encode(x='x', y='y', color=alt.value("#FFAA00"))
            pred = alt.Chart(source).mark_line(opacity=0.6).encode(x='x', y='z', color=alt.value("#FF00BB"))
            both = alt.layer(real, pred)
            self.the_plot.altair_chart(both,use_container_width=True)
            self.prev_score = bs



    def ktsne(self,x, pdim=4 , kernel='pca'):
        print (x.shape)
        f_opts = {'p_degree': 1.0, 'p_dims': pdim, 'mode': 'CLA', 'eta': 25.0, 
            'perplexity': 20.0, 'n_dims': 2, 'ker': kernel, 'gamma': 1.0}        
            
        m_tsne = Mtsne(x, f_opts=f_opts)
        X_reduced = m_tsne.get_solution(1500)
        X_reduced = self.scaler.fit_transform(X_reduced)
        return X_reduced


    def getXY(self,dt="iris"):
        if dt == "iris":
            X,y = datasets.load_iris(return_X_y=True)
        else:
            X,y  = datasets.load_digits(return_X_y=True)
        return X, y

    def classify(self):


        nrOpts = {  "opx": 1, "depth": 2, "nvars": None, "pvc": .8, "pf": 1., 
                    "cross": .6, "mut": .2, "mrand": 1.}
        gOpts = {   "mxepoch": self.epochs, "bsize": 500, "bupdate": 50, "fraction": .8,
                    "history": 5, "mxtries": 15, "mode": 'CLA'}                


        print(nrOpts)
        print(gOpts)
        dset = 'iris'
        X, y = self.getXY(dset)

        X, y = shuffle(X, y)
        o = np.unique(y).size

        X = X[:500]
        y = y[:500]

        time.sleep(2)
        # X0 = PCA(n_components=2).fit_transform(X)
        X0 = self.ktsne(X, pdim=20 , kernel='pca')
        Xtr = X0[:-10]
        Xts = X0[-10:]
        ytr = y[:-10]
        yts = y[-10:]
        i = Xtr.shape[1]
        self.datas = {"X": Xtr, "Y": ytr, "nX": Xts, "nY": yts}

        prg_bar = st.empty()
        st.subheader(f' {dset} Real: ')

        self.plot_real_cla(Xtr, ytr)
        time.sleep(20)
        st.subheader(f' {dset}  Predicted ')

        self.the_plot = st.empty() 
        
        self.prev_score = np.inf
        l_dims = [i, o]

        self.net = NClassifier(dim=l_dims, datas=self.datas, nr_opts=nrOpts, g_opts=gOpts)
        self.net.setLoading(True)
        lf = PThread(target=self.net.train)
        st.report_thread.add_report_ctx(lf)
        lf.start()

        while(True):
            time.sleep(5.)
            self.cla_thread()
            e,m = self.net.get_cur_epoc()-1 , self.net.get_max_epoc()
            prg_bar.text(f' peogress: {np.round(100*e/m,2)}% ')
            on = self.net.getLoading()
            if not on:
                break


    def plot_real_cla(self, X, Y):
            x , y = X[:,0].ravel(),X[:,1].ravel()
        
            source = pd.DataFrame({'x':x , 'y': y , 'z': Y*50. },columns=['x','y','z'])
            pred = alt.Chart(source).transform_calculate(
                x1=alt.datum.x - 0.5,
                x2=alt.datum.x + 0.5,
                y1=alt.datum.y - 0.5,
                y2=alt.datum.y + 0.5,
            ).mark_circle(size=100).encode(
                x='x1:Q', x2='x2:Q',
                y='y1:Q', y2='y2:Q',
                color = alt.Color('z:Q', scale=alt.
                      Scale(scheme = 'dark2')  ))
            st.altair_chart(pred,use_container_width=True)
            


    def plot_cla(self, X, Y, yh):
        bs = self.net.get_best_score()
        if (  bs < self.prev_score  ):
            x , y = X[:,0].ravel(),X[:,1].ravel()
        
            source = pd.DataFrame({'x':x , 'y': y , 'z': yh },columns=['x','y','z'])
            pred = alt.Chart(source).transform_calculate(
                x1=alt.datum.x - 0.5,
                x2=alt.datum.x + 0.5,
                y1=alt.datum.y - 0.5,
                y2=alt.datum.y + 0.5,
            ).mark_circle(size=100).encode(
                x='x1:Q', x2='x2:Q',
                y='y1:Q', y2='y2:Q',
                color = alt.Color('z:Q', scale=alt.
                      Scale(scheme = 'dark2')  ))

            self.the_plot.altair_chart(pred,use_container_width=True)
            self.prev_score = bs


    def cla_thread(self, *largs,**kwargs):
        X , Yh = self.net.getYhat(nx=False,multi=False)
        Y = self.datas["Y"]
        self.plot_cla(X, Y, Yh)
    

    def reg_thread(self, *largs,**kwargs):
        X , Yh = self.net.getYhat(nx=False,multi=False)
        Y = self.datas["Y"]
        self.plot_reg(X, Y, Yh)


