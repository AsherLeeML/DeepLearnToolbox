function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)

        dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];

        switch nn.optim
            case 'rmsprop'
                nn.rW{i} = 0.99 * nn.rW{i} + 0.01 * dW.^2;
                dW = nn.learningRate * dW ./ (sqrt(nn.rW{i}) + nn.epsilon);
            case 'adam'
                nn.adam.t  = nn.adam.t + 1;
                nn.adam.m{i}  = nn.adam.beta1 * nn.adam.m{i} + (1 - nn.adam.beta1) * dW;
                mt = nn.adam.m{i} / (1-nn.adam.beta1^nn.adam.t);
                nn.adam.v{i}  = nn.adam.beta2 * nn.adam.v{i} + (1-nn.adam.beta2)*(dW.^2);
                vt = nn.adam.v{i} / (1-nn.adam.beta2^nn.adam.t);
                dW = nn.learningRate * mt ./ (sqrt(vt) + nn.adam.epsilon);
        end
                
        
            
        nn.W{i} = nn.W{i} - dW;
    end
    
    if nn.batch_norm
        for i = 1:(nn.n - 2)
            nn.rBN{i} = 0.9 * nn.rBN{i} + 0.1 * nn.dBN{i} .^ 2;
            dBN = nn.learningRate * nn.dBN{i} ./ (sqrt(nn.rBN{i}) + nn.epsilon);
            nn.vBN{i} = nn.momentum * nn.vBN{i} + dBN;
            nn.gamma{i} = nn.gamma{i} - nn.vBN{i}(1:length(nn.gamma{i}));
            nn.beta{i} = nn.beta{i} - nn.vBN{i}(length(nn.gamma{i})+1:end);
        end
    end
    
    if strcmp(nn.activation_function, 'relu')
        da = nn.learningRate * nn.da;
        nn.va = 0.5*nn.va + da;
        nn.ra = nn.ra - nn.va;
    end
end
