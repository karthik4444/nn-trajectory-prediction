import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
from scipy.stats import multivariate_normal

'''
Need to determine proper hyper parameters for training

Training With
- 1 GRU Layer
- 0 embedding layers
- hidden layer dimension = 128
- default: no BPTT truncation  
- GPU optimized

Input Layer >>> GRU layer #>> Output Layer

'''

class SimpleGRU:
    def __init__(self, input_dim, output_dim, hidden_dim=128, bptt_truncate=-1):
        #instance variables for LSTM
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.output_dim = output_dim
        #network parameters
        optim_range = np.sqrt(1./hidden_dim)
        U = np.random.uniform(-1 * optim_range, optim_range, (3, hidden_dim, input_dim))
        W = np.random.uniform(-1 * optim_range, optim_range, (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-1 * optim_range, optim_range, (output_dim, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(output_dim)
        #shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        #RMSProp parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        #building the theano computational graph
        self.__theano_build()

    def __theano_build(self):
        U, W, V, b, c = self.U, self.W, self.V, self.b, self.c
        x = T.fmatrix('x')
        y = T.fmatrix('y')
        k = T.iscalar('k')

        def forward_prop_step(x_t, s_prev):
            #Embedding Layer
            x_e = x_t
            # GRU Layer 1
            z_t = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_prev) + b[1])
            c_t = T.tanh(U[2].dot(x_e) + W[2].dot(s_prev * r_t) + b[2])
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
        [preds, s2], updates2 = theano.scan(
            forward_prop_step,
            n_steps=k,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=x[-1].astype(theano.config.floatX)), dict(initial=s1[-1].astype(theano.config.floatX))]
            )
        loss = T.mean((preds - y) ** 2)

        #back-propogation through time. Truncation is handled upon calculating o.
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
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        #1e-6 gaurds against division by 0
        #gradient descent update of parameters
        self.sgd_step = theano.function(
            [x, y, k, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[(U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
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

        self.predict = theano.function([x], next_step)
        self.loss = theano.function([x, y, k], loss)

        def cost(self, X, Y, num_training_examples):
            #average loss = cost
            return (np.sum([self.loss(x,y) for x,y in zip(X,Y)])) / num_training_examples


model = SimpleGRU(2, 2, 128)