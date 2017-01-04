import numpy as np
import theano as theano
import theano.tensor as T

class PoolingGRU:
    def __init__(self, input_dim, output_dim, pooling_size, hidden_dim=128, bptt_truncate=-1):
        #instance variables for GRU
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.output_dim = output_dim
        #network parameters
        optim_range = np.sqrt(1./hidden_dim)

        #Embedding Matrices
        D = np.random.uniform(-1 * optim_range, optim_range, (hidden_dim, pooling_size * pooling_size * hidden_dim))
        E = np.random.uniform(-1 * optim_range, optim_range, (hidden_dim, input_dim))

        U = np.random.uniform(-1 * optim_range, optim_range, (3, hidden_dim, hidden_dim * 2))
        W = np.random.uniform(-1 * optim_range, optim_range, (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-1 * optim_range, optim_range, (output_dim, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(output_dim)
        #shared variables
        self.D = theano.shared(name='D', value=D.astype(theano.config.floatX))
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))

        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        #RMSProp parameters
        self.mD = theano.shared(name='mD', value=np.zeros(D.shape).astype(theano.config.floatX))
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        #building the theano computational graph
        self.__theano_build()

    def __theano_build(self):
        D, E, U, W, V, b, c = self.D, self.E, self.U, self.W, self.V, self.b, self.c
        x = T.fmatrix('x')
        y = T.fvector('y')
        H = T.ftensor4('H')

        xt = T.fvector('xt')
        Ht = T.ftensor3('Ht')
        s_prev = T.fvector('s_prev')

        def ReLU(x):
            return T.switch(x<0, 0, x)

        def time_step(H_t, x_t, s_prev):
            #Embedding Layer. Hidden pooling tensor is flattened, and embedded into vector with ReLU non-linearity
            #embedded hidden pooling tensor is concatenated to embedded input vector
            H_e = ReLU(D.dot(H_t.flatten(1)))
            x_e = ReLU(E.dot(x_t))
            i = T.concatenate([x_e, H_e])

            # GRU Layer
            z_t = T.nnet.hard_sigmoid(U[0].dot(i) + W[0].dot(s_prev) + b[0])
            r_t = T.nnet.hard_sigmoid(U[1].dot(i) + W[1].dot(s_prev) + b[1])
            c_t = ReLU(U[2].dot(i) + W[2].dot(s_prev * r_t) + b[2])
            s_t = (T.ones_like(z_t) - z_t) * c_t + z_t * s_prev
            #prediction at time t+1
            o_t = V.dot(s_t) + c

            return [o_t, s_t]
        
        np, nh = time_step(Ht, xt, s_prev)
        self.time_step = theano.function([xt, Ht, s_prev], [np, nh], allow_input_downcast=True)

        #feed-forward for training example.
        #initializing the hidden state with first 8 steps
        [o, s1], updates1 = theano.scan(
            time_step,
            sequences=[H, x],
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))]
            )
        self.predict = theano.function([x, H], o[-1], allow_input_downcast=True)

        #loss defined by square distance between predicted and actual
        loss = T.dot(o[-1] - y, o[-1] - y)
        self.loss = theano.function([x, H, y], loss, allow_input_downcast=True)


        #back-propogation through time. Truncation is handled upon calculating o.
        dD = T.grad(loss, D)
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
        mD = decay * self.mD + (1 - decay) * dD ** 2
        mE = decay * self.mE + (1 - decay) * dE ** 2
    
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        #1e-6 gaurds against division by 0
        #gradient descent update of parameters
        self.sgd_step = theano.function(
            [x, H, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            allow_input_downcast=True,
            updates=[
                    (D, D - learning_rate * dD / T.sqrt(mD + 1e-6)),
                    (E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                    (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                    (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                    (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                    (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                    (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                    (self.mD, mD),
                    (self.mE, mE),
                    (self.mU, mU),
                    (self.mW, mW),
                    (self.mV, mV),
                    (self.mb, mb),
                    (self.mc, mc)
                    ])
