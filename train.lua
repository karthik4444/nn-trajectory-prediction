require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

local GRU = require 'model.GRU'


opt = {
    input_size = 2
    data_dir = '' -- MUST BE DETERMINED
    hidden_layer_size = 128,
    num_layers = 2,
    learning_rate = 0.003,
    learning_rate_decay = 0.97,
    learning_rate_decay_after = 10,
    decay_rate = 0.95,
    dropout = 0.5,
    seq_length = 50,
    batch_size = 50,
    max_epochs = 50,
    grad_clip = 10,
    seed = 123,
    print_every = 1,
    eval_val_every = 1000,
    checkpoint_dir = 'cv'
    savefile = 'gru_model',
    gpuid = 1
}

torch.setnumthreads(2)
torch.manualSeed(opt.seed)
protos = {}
print('creating gru model with' .. opt.num_layers .. 'layers')
protos.gru = GRU.init(opt.input_size, opt.hidden_layer_size, opt.dropout)
init_state = {}
for L=1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cudo() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
protos.criterion = nn.MSECriterion()
params, grad_params = protos.gru:getParameters()
params:uniform(-0.08, 0.08) --Maybe change to 1/sqrt(n)
print('number of parameters in the model: ' .. params:nElement())

clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = crnn.clone
end

