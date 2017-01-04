import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
'''
Need to determine proper hyper parameters for training

Training With
- 1 GRU Layer
- hidden layer dimension = 128
- default: no BPTT truncation  
- ready for GPU use

'''

class NaiveGRU:
    #initialize network parameters
    def __init__(self, input_dim, output_dim, hidden_dim=128, bptt_truncate=-1):
        #instance variables for GRU
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.output_dim = output_dim
        #network parameters
        optim_range = np.sqrt(1./hidden_dim)
        E = np.random.uniform(-1 * optim_range, optim_range, (hidden_dim, input_dim))
        U = np.random.uniform(-1 * optim_range, optim_range, (3, hidden_dim, hidden_dim))
        W = np.random.uniform(-1 * optim_range, optim_range, (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-1 * optim_range, optim_range, (output_dim, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(output_dim)
        #shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        #RMSProp parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        #building the theano computational graph
        self.__theano_build()

    def __theano_build(self):
        E, U, W, V, b, c = self.E, self.U, self.W, self.V, self.b, self.c
        x = T.fmatrix('x')
        y = T.fvector('y')

        #implementation of ReLU activator
        def ReLU(x):
            return T.switch(x<0, 0, x)

        def forward_prop_step(x_t, s_prev):
            #Embedding Layer with ReLU non-linearity
            x_e = ReLU(E.dot(x_t))
            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_prev) + b[1])
            c_t = ReLU(U[2].dot(x_e) + W[2].dot(s_prev * r_t) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_prev
            #prediction at time t+1
            o_t = V.dot(s_t) + c

            return [o_t, s_t]

        #feed-forward for training example.
        #initializing the hidden state with first 8 steps
        [o, s1], updates1 = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))]
            )

        #using first 8 steps to predict the future trajectory
        loss = T.dot((o[-1] - y), (o[-1] - y))



        #back-propogation through time. Truncation is handled upon calculating o.
        dE = T.grad(loss, E)
        dU = T.grad(loss, U)
        dW = T.grad(loss, W)
        db = T.grad(loss, b)
        dV = T.grad(loss, V)
        dc = T.grad(loss, c)


        #Stochastic Gradient Descent
        #sgd parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')



        #RMSProp updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        #1e-6 gaurds against division by 0
        #gradient descent update of parameters
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            allow_input_downcast=True,
            updates=[
                    (E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                    (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                    (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                    (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                    (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                    (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                    (self.mU, mU),
                    (self.mW, mW),
                    (self.mV, mV),
                    (self.mb, mb),
                    (self.mc, mc)
                    ])

        self.predict = theano.function([x], o[-1], allow_input_downcast=True)
        self.loss = theano.function([x, y], loss, allow_input_downcast=True)

        def cost(X, Y):
            return (np.sum([self.loss(x,y) for x,y in zip(X,Y)])) / len(X)
        self.cost = cost
