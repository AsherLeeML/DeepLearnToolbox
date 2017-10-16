function nn = nnsetup(architecture, config)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    if nargin > 1
       nn.activation_function               = config.act;   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
        nn.learningRate                     = config.lr;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
        nn.scaling_learningRate             = config.scaleRate;            %  Scaling factor for the learning rate (each epoch)
        nn.weightPenaltyL2                  = config.reg;            %  L2 regularization
        nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
        nn.sparsityTarget                   = 0.05;         %  Sparsity target
        nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
        nn.dropoutFraction                  = config.dropout;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
        nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
        nn.output                           = config.output;       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'\
        nn.batch_norm                       = config.batch_norm;
        nn.epsilon                          = 1e-10;
        nn.hinge_norm                       = config.hinge_norm;
        nn.optim                            = config.sgd;
    else
         % default configration
        nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
        nn.learningRate                     = 0.001;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
        nn.scaling_learningRate             = 1;            %  Scaling factor for the learning rate (each epoch)
        nn.weightPenaltyL2                  = 0;            %  L2 regularization
        nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
        nn.sparsityTarget                   = 0.05;         %  Sparsity target
        nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
        nn.dropoutFraction                  = 0.5;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
        nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
        nn.output                           = 'softmax';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'\
        nn.batch_norm                       = 1;
        nn.epsilon                          = 1e-10;
        nn.hinge_norm                       = 1;
        nn.optim                             = 'adam';
    end

    for i = 2 : nn.n   
        
        nn.ra(i-1) = 0.2;
        nn.va(i-1) = 0;
        nn.pow(i-1) = 0.5;
        nn.vpow(i-1) = 0;
        
        % weights and weight momentum
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        nn.rW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % for adam
        nn.adam.t       = 0;
        nn.adam.beta1   = 0.9;
        nn.adam.beta2   = 0.999;
        nn.adam.epsilon = 1e-8;
        nn.adam.m{i-1}       = zeros(size(nn.W{i-1}));
        nn.adam.v{i-1}       = zeros(size(nn.W{i-1}));
        
        % for batch_norm
        nn.beta{i - 1} = zeros(1, nn.size(i));
        nn.gamma{i - 1} = ones(1, nn.size(i));
        nn.sigma2{i - 1} = ones(1, nn.size(i));
        nn.mu{i - 1} = zeros(1, nn.size(i));
        nn.vBN{i - 1} = zeros(1, nn.size(i) * 2);
        nn.rBN{i - 1} = zeros(1, nn.size(i) * 2);
        nn.mean_sigma2{i - 1} = zeros(1, nn.size(i));
        nn.mean_mu{i - 1} = zeros(1, nn.size(i));
        
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
    nn.ra = nn.ra(1:end-1);
    nn.va = nn.va(1:end-1);
end
