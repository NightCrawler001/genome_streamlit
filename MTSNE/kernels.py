
import numpy as np

class Kernels(object):
    def __init__(self, XX , k_opts= { "kernel" : "pca" , "gamma": .5, "degree" : 1 , "pcomp": 4 }):
        self.X = XX.copy()
        self.k_opts = k_opts



    def process_data(self) :
        gamma = self.k_opts["gamma"]
        p_degree = self.k_opts["degree"]
        p_comp = self.k_opts["p_dims"]
        ker = self.k_opts["kernel"]
        XX = self.X
        # pca cosine anova sigmoid iquad=student cauchy ..
        if ker == "poly" :
            X = self.poly(XX, gamma=gamma ,degree=p_degree , n_components = p_comp   ).real
        elif ker == "anova" :
            X = self.anova(XX, gamma=gamma ,degree=p_degree , n_components = p_comp   ).real
        elif ker == "student" :
            X = self.t_student( XX , degree=p_degree , n_components=p_comp ).real
        elif ker == "rbf" :
            X = self.rbf( XX , gamma=gamma ,  n_components=p_comp ).real
        elif ker == "sigmoid" :
            X = self.sig( XX , gamma=gamma ,r=1 , n_components=p_comp ).real
        elif ker == "cosine" :
            X = self.cosine( XX , n_components=p_comp ).real
        elif ker == "quad" :
            X = self.quad( XX , gamma= gamma ,degree=p_degree , n_components=p_comp ).real
        elif ker == "iquad" :
            X = self.iquad( XX , gamma=gamma ,degree=p_degree , n_components=p_comp ).real
        elif ker == "cauchy" :
            X = self.cauchy( XX , gamma=gamma , n_components=p_comp ).real
        elif ker == "fourier" :
            X = self.fourier( XX , gamma = gamma , n_components=p_comp ).real
        elif ker == "linear" :
            X = self.linear( XX , gamma=1 ,degree=1 , n_components=p_comp ).real
        else:
            X = self.pca(XX , n_components=p_comp).real

        return X


    def eignes(self, M , n_components=4):
        (vals, V) = np.linalg.eig(M)
        idx = vals.argsort()[::-1]
        vals = vals[idx].real
        print("---------------- vals: ----------------")
        print(n_components , vals.shape)
        print(vals[:n_components]) 
        print('---------------------------------------') 
        V = V[:,idx]
        U = V[:,0:n_components]
        return U


    def pca(self, X , n_components=2):

        (n, d) = X.shape
        X -= np.mean(X, 0)
        _cov = np.cov(X.T)
        U = self.eignes(_cov,n_components=n_components)
        XX = np.dot(X, U)
        return XX

    def linear(self,X, gamma=1, degree=1, n_components=2):
        # Calculating kernel
        X -= np.mean(X, 0)
        K = (gamma*X.dot(X.T))**degree

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)


        return self.eignes(K,n_components=n_components)


    def poly(self,X, gamma=1, degree=2, n_components=2):
        # Calculating kernel
        X -= np.mean(X, 0)
        K = (gamma*X.dot(X.T)+1)**degree

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)


        return self.eignes(K,n_components=n_components)

    def rbf(self,X, gamma=.1, n_components=2):

        # Calculating the squared Euclidean distances for every pair of points
        # in the MxN dimensional dataset.
        X -= np.mean(X, 0)
        print("heur gamma ?  : " , 1./X.shape[1])
        mat_sq_dists = np.sum( (X[None,:] - X[:, None])**2, -1)
        K=np.exp(-gamma*mat_sq_dists)

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)

        return self.eignes(K,n_components=n_components)



    def sig(self,X, gamma=.2, r=1, n_components=2):

        X -= np.mean(X, 0)
        K = self.sigmoid_l(gamma* X.dot(X.T) + r)

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)

        return self.eignes(K,n_components=n_components)



    def cosine(self, X , n_components=2):
        #self._dim = data_1.shape[1]
        X -= np.mean(X, 0)
        
        norm_1 = ((X ** 2).sum(axis=1)).reshape(X.shape[0], 1)
        #norm_2 = np.sqrt((X ** 2).sum(axis=1)).reshape(X.shape[0], 1)
        K = X.dot(X.T) / (norm_1)

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)
        
        return self.eignes(K,n_components=n_components)


    def quad(self , X ,gamma=30 , degree=5 ,  n_components=2 ):

        X -= np.mean(X, 0)

        dists_sq = np.sum( (X[None,:] - X[:, None])**2, -1)
        K = 1. - (dists_sq / (dists_sq + gamma)**degree )
        
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)

        return self.eignes(K,n_components=n_components)


    def iquad(self , X ,gamma=1 , degree=.5 ,n_components=2 ):

        X -= np.mean(X, 0)
        
        degree = 1. # better small degree
        
        dists_sq = np.sum( (X[None,:] - X[:, None])**2, -1)
        K = 1. / (dists_sq + gamma**2)**degree

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)



        return self.eignes(K,n_components=n_components)

    def cauchy(self , X , gamma=.2 ,n_components=2 ):

        X -= np.mean(X, 0)
        dists_sq = np.sum( (X[None,:] - X[:, None])**2, -1)
        K = 1 / (1 + dists_sq*gamma)

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)

        return self.eignes(K,n_components=n_components)


    def t_student(self , X , degree=2 , n_components=2 ):

        X -= np.mean(X, 0)
        dists_sq = np.sum( (X[None,:] - X[:, None])**2, -1)
        K = self.student(dists_sq,a=degree)

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)


        return self.eignes(K,n_components=n_components)


    def anova(self , X , gamma=.01 , degree=1 , n_components=2 ):

        X -= np.mean(X, 0)
        K = np.zeros((X.shape[0], X.shape[0]))
        for d in range(X.shape[1]):
            column_1 = X[:, d].reshape(-1, 1)
            K += np.exp( -gamma * (column_1 - column_1.T)**2 ) ** degree


        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)


        return self.eignes(K,n_components=n_components)


    def fourier(self , X , gamma=.1 , n_components=2 ):

        X -= np.mean(X, 0)
        K = np.ones((X.shape[0], X.shape[0]))
        gamma = min(.1,gamma)
        for d in range(X.shape[1]):
            column_1 = X[:, d].reshape(-1, 1)
            K *= (1-gamma ** 2) / (2*(1 - 2*gamma *np.cos(column_1 - column_1.T)) + gamma**2)

        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        print(X.shape , K.shape)



        return self.eignes(K,n_components=n_components)



    def student(self,x,a=1):
        return 1/(1+np.abs(x)**a)


    def sigmoid_l(self,x,a=0,b=1):

        s = 1.0/(1+np.exp(b*(a-x)))
        return  s


    def sigmoid_r(self,x,a=0,b=1):

        s = 1.0/(1+np.exp(b*(x-a)))
        return  s
