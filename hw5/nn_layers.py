import numpy as np
from skimage.util.shape import view_as_windows

def convolution(x, filter):
        window = view_as_windows(x,filter.shape)
        return np.tensordot(window, filter)

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        out_h = x.shape[2] + 1 - self.W.shape[2]
        out_w = x.shape[3] + 1 - self.W.shape[3]
        out = np.zeros((x.shape[0], self.W.shape[0], out_h , out_w))
        for batch in range(x.shape[0]):
            for ch in range(self.W.shape[1]):
                window = view_as_windows(x[batch,ch,:,:], (self.W.shape[2],self.W.shape[3]))
                for filter_num in range(self.W.shape[0]):
                    out[batch,filter_num,:,:] += np.tensordot(window, self.W[filter_num][ch])
        for batch in range(x.shape[0]):
            out[batch,:,:,:] += self.b[0,:,:,:]

        return out

    def backprop(self, x, dLdy):
        dLdx = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        for batch in range(x.shape[0]):
            for ch in range(x.shape[1]):
                for filter_num in range(dLdy.shape[1]):
                    dLdx[batch,ch,:,:] += convolution(np.pad(dLdy[batch,filter_num,:,:],((2,2),(2,2)),'constant',constant_values=0 ), np.rot90(self.W[filter_num,ch,:,:],2))
        dLdW = np.zeros((dLdy.shape[1], x.shape[1] , self.W.shape[2], self.W.shape[3]))
        for batch in range(x.shape[0]):
            for ch in range(x.shape[1]):
                window = view_as_windows(x[batch,ch,:,:], (dLdy.shape[2],dLdy.shape[3]))
                for filter_num in range(dLdy.shape[1]):
                    dLdW[filter_num,ch,:,:] += np.tensordot(window, dLdy[batch][filter_num])
        dLdb = np.zeros(self.b.shape)
        for filter_num in range(dLdb.shape[1]):
            dLdb[0][filter_num][0][0] += np.sum(dLdy[:,filter_num,:,:])
        return dLdx, dLdW, dLdb

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        self.recentOut = np.zeros([0])
        self.recentOutArg = np.zeros([0])

    def forward(self, x):
        out_h = int(x.shape[2]/2)
        out_w = int(x.shape[3]/2)
        out = np.zeros((x.shape[0],x.shape[1],out_h, out_w))
        outarg = np.zeros((x.shape[0],x.shape[1],out_h, out_w))
        for batch in range(x.shape[0]):
            for ch in range(x.shape[1]):
                window = view_as_windows(x[batch][ch], (self.pool_size,self.pool_size), step = self.stride)
                rewin = window.reshape(window.shape[0],window.shape[1],window.shape[2]*window.shape[3])
                out[batch][ch] = np.max(rewin, axis=2)
                outarg[batch][ch] = np.argmax(rewin, axis=2)
                        
        self.recentOut = out
        self.recentOutArg = outarg
        return out

    def backprop(self, x, dLdy):
        redLdy = np.reshape(dLdy, (x.shape[0],x.shape[1],int(x.shape[2]/2),int(x.shape[3]/2)))
        dLdx = np.zeros(x.shape)
        for batch in range(x.shape[0]):
            for ch in range(x.shape[1]):
                for h in range(redLdy.shape[2]):
                    for w in range(redLdy.shape[3]):
                         dLdx[batch][ch][h+int(self.recentOutArg[batch][ch][h][w]/2)][w+int(self.recentOutArg[batch][ch][h][w]%2)] = redLdy[batch][ch][h][w]
                            
        return dLdx



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        rex = np.reshape(x,(x.shape[0], -1))
        return (self.W@rex.T).T+np.tile(self.b,(x.shape[0],1))

    def backprop(self,x,dLdy):
        rex = np.reshape(x,(x.shape[0],-1))
        #print(dLdy)
        dLdW = dLdy.T@rex
        #print(dLdW)
        dLdx = dLdy@self.W
        #print(dLdx)
        dLdb = np.sum(dLdy, axis=0,keepdims=True)

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

def step_function(x):
    return np.array(x>0, dtype=np.int)

def Relu(x):
    return step_function(x)*x

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        return Relu(x)
    
    def backprop(self, x, dLdy):
        dLdx = step_function(x)*dLdy
        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        self.old_y = np.zeros([0])
        pass

    def forward(self, x):
        max = np.max(x,axis=1,keepdims=True)
        eX = np.exp(x-max)
        sum = np.sum(eX,axis=1, keepdims=True)
        self.old_y = eX/sum
        return self.old_y

    def backprop(self, x, dLdy):
        arr = np.zeros([x.shape[0],x.shape[1]])
        for batch_size in range(x.shape[0]):
            local = np.diag(self.old_y[batch_size]) - np.array([self.old_y[batch_size]]).T.dot(np.array([self.old_y[batch_size]]))
            arr[batch_size,:] = dLdy[batch_size]@local
        
        return arr
##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########


class nn_cross_entropy_layer:

    def __init__(self):
        self.label=np.zeros([0])

    def forward(self, x, y):
        delta = 1e-7
        arr = np.zeros(x.shape)
        for batch_size in range(x.shape[0]):
            arr[batch_size][y[batch_size]]=1
        self.label = arr
        return  -np.sum(arr*np.log(x+delta))/x.shape[0]

    def backprop(self, x, y):
        delta = 1e-7
        return -1*self.label/(x*x.shape[0])