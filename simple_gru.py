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
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, input_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        P = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (5, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(output_dim)
        #shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.P = theano.shared(name='P', value=P.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        #RMSProp parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mP = theano.shared(name='mP', value=np.zeros(P.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        #building the theano computational graph
        self.__theano_build()

    def __theano_build(self):
        U, W, V, P, b, c = self.U, self.W, self.V, self.P, self.b, self.c
        x = T.fvector('x')
        y = T.fvector('y')

        def forward_prop_step(x_t, s_t1prev, s_t2_prev):
            #Embedding Layer
            x_e = x_t
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            #prediction at time t+1
            o_t1 = V[0].dot(s_t1) + c[0]

            return [o_t1, s_t1, s_t2]

        #feed-forward for training example.
        [o, s1, s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim)), dict(initial=T.zeros(self.hidden_dim)), None]
            )

        #loss function (negative log liklehood)
        #gaussian_param formatted as [sigma1, sigma2, std1, std2, rho]
        errors = []
        pred, traj1, traj2 = o, s1, s2
        for point in y:
            gaussian_param = P.dot(traj2)
            M = [gaussian_param[0], gaussian_param[1]]
            V = [[gaussian_param[2]**2, gaussian_param[2] * gaussian_param[3] * gaussian_param[4]],
                [gaussian_param[2] * gaussian_param[3] * gaussian_param[4], gaussian_param[3]**2]]
            error = multivariate_normal.logpdf([point[0], point[1]], mean=M, cov=V)
            errors.append(error)
            pred, traj1, traj2 = forward_prop_step(pred, traj1, traj2)

        loss = - T.sum(errors)

        #back-propogation through time. Truncation is handled upon calculating o.
        dU = T.grad(loss, U)
        dW = T.grad(loss, W)
        dP = T.grad(loss, P)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        #Stochastic Gradient Descent
        #sgd parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        #RMSProp updates
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mP = decay * self.mP + (1 - decay) * dP ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        #1e-6 gaurds against division by 0
        #gradient descent update of parameters
        update_params = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (P, P - learning_rate * dP / T.sqrt(mP + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mP, mP),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

        self.sgd_step = update_params
        self.predict = theano.function([x], next_step)
        self.loss = theano.function([x, y], loss)

        def cost(self, X, Y, num_training_examples):
            #average loss = cost
            return (np.sum([self.loss(x,y) for x,y in zip(X,Y)])) / num_training_examples