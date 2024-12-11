import numpy as np
import time

class NeuralNetwork():
    def __init__(self,size):
        self.size = size
        self.params = {}
        for i in range(len(size)-1):
            self.params["W" + str(i+1)]=np.random.randn(size[i+1], size[i]) * np.sqrt(2. / size[i])
            self.params["B" + str(i+1)]=np.zeros((size[i+1], 1))
        self.cache = {}

    @staticmethod
    def sigmoid(n,deriv=False):
        x = 1 / (1 + np.exp(-n))
        if deriv:
            return x * (1 - x)
        return x

    @staticmethod
    def softmax(n):
        return np.exp(n) / np.sum(np.exp(n),axis=0)

    def feedforward(self,A0):
        self.cache["A0"] = A0
        for i in range(1,len(self.size)):
            iStr = str(i)
            self.cache["Z" + iStr] = self.params["W" + iStr].dot(self.cache["A" + str(i - 1)]) + self.params["B" + iStr]
            self.cache["A" + iStr] = self.sigmoid(self.cache["Z" + iStr])

    @staticmethod
    def getpred(A2):
        return np.argmax(A2,0)

    def backprop(self,DESIRED):
        dWs = []
        dBs = []
        m = DESIRED.size
        DESIRED = np.eye(self.size[len(self.size)-1])[DESIRED].T
        sizeLen = len(self.size)-1
        self.cache["dZ" + str(sizeLen)] = self.cache["A" + str(sizeLen)] - DESIRED
        dBs.append(1 / m * np.sum(self.cache["dZ" + str(sizeLen)],axis=1,keepdims=True))
        dWs.append(1 / m * self.cache["dZ" + str(sizeLen)].dot(self.cache["A" + str(sizeLen-1)].T))
        for i in reversed(range(1,sizeLen)):
            dZ = self.params["W" + str(i+1)].T.dot(self.cache["dZ" + str(i+1)]) * self.sigmoid(self.cache["Z" + str(i)],True)
            dWs.append(1 / m * dZ.dot(self.cache["A" + str(i-1)].T))
            dBs.append(1 / m * np.sum(dZ,axis=1,keepdims=True))
            self.cache["dZ" + str(i)] = dZ
        dWs.reverse()
        dBs.reverse()
        return dWs,dBs

    def updateparams(self,dWs,dBs,lr):
        for i in range(1,len(dWs)+1):
            self.params["W" + str(i)] -= lr * dWs[i-1]
            self.params["B" + str(i)] -= lr * dBs[i-1]

    def gradientdescent(self,A0,DESIRED,lr,ITERS):
        start = time.time()
        for i in range(ITERS):
            self.feedforward(A0)
            dWs,dBs = self.backprop(DESIRED)
            self.updateparams(dWs,dBs,lr)
            if i+1 % 50 == 0:
                print(i,"iters done.")
        print(f"Time elapsed:{time.time()-start}")
