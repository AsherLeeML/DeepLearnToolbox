function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;

    %feedforward pass
    for i = 2 : n-1
        if nn.batch_norm
            if nn.testing
                nn.a_pre{i} = nn.a{i-1} * nn.W{i-1}';
                norm_factor = nn.gamma{i-1} ./ sqrt(nn.mean_sigma2{i-1} + nn.epsilon);
                nn.a_hat{i} = bsxfun(@times, nn.a_pre{i}, norm_factor);
                nn.a_hat{i} = bsxfun(@plus, nn.a_hat{i}, nn.beta{i-1} - norm_factor .* nn.mean_mu{i-1});
            else
                nn.a_pre{i} = nn.a{i-1} * nn.W{i-1}';
                nn.mu{i-1} = mean(nn.a_pre{i});
                x_mu = bsxfun(@minus, nn.a_pre{i}, nn.mu{i-1});
                nn.sigma2{i-1} = mean(x_mu .^ 2);
                norm_factor = nn.gamma{i-1}./sqrt(nn.sigma2{i-1}+nn.epsilon);
                nn.a_hat{i} = bsxfun(@times, nn.a_pre{i}, norm_factor);
                nn.a_hat{i} = bsxfun(@plus, nn.a_hat{i}, nn.beta{i-1} - norm_factor .* nn.mu{i-1});
            end
        else
            nn.a_hat{i} = nn.a{i-1} * nn.W{i-1}';
        end
        
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
            case 'relu'
                nn.a{i} = max(nn.a_hat{i}, 0) + nn.ra(i-1) * min(nn.a_hat{i}, 0);
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
    switch nn.output 
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
        case 'relu'
            nn.a{n} = max(nn.a{n-1} * nn.W{n-1}', 0);
        case 'hinge'
            nn.a{n} = nn.a{n-1} * nn.W{n-1}';
            if ~nn.testing
                nn.e = 2*(y-0.5);
                nn.a{n} = max(0, 1-nn.a{n} .* nn.e);
            end
    end

    %error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
        case 'hinge'
            if nn.hinge_norm
                nn.L = sum(sum(abs(nn.a{n}))) / m;
            else
                nn.L = 1/2 * sum(sum(nn.a{n} .* nn.a{n})) /m;
            end
    end
end
