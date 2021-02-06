function [Ws, a, b, errors] = training(X, Ws, a, b, cd_k, eta, alpha, alpha_end, lambda, batch_size, max_epochs)
    % %% Support variables
    count = 0;       % iteration count, for interrupting learning when count>max_epoch
    errors = [Inf];  % initialize error vector

    Nh = size(b, 1); % hidden layer neurons
    Ni = size(a, 1); % input layer neurons
    Nd = size(X, 1); % number of data samples

    % batch data and shuffle them
    n_batches = ceil(Nd/batch_size);
    %batchdata = zeros(batch_size, Ni, n_batches);
    
    % %% Training process - epoch computation
    while true
        
        % initialize updates with proper dimentions
        deltaW = zeros(Ni, Nh);
        deltaa = zeros(Ni, 1);
        deltab = zeros(Nh ,1);
        
        error = 0;
        
        % shuffle inputs
        fprintf('-- Shuffle inputs\n');
        X = X(randperm(size(X, 1)),:);
        
        % stochastic updates
        fprintf('-- Training...\n');
        
        for i = 1:n_batches
            % minibatch definition
            start = (i-1)*batch_size+1;
            finish = min(batch_size*i, Nd);
            minibatch = X(start:finish, :);

            % CD-k step, computed on minibatch
            %fprintf('-- CD-1 step...\n');
            [h0, v0, vk, hk] = rbm_CD_k(Ws, a, b, cd_k, minibatch');
            
            % gradient computation 
            %fprintf('-- Gradient computation...\n');
            [Wgrad, agrad, bgrad] = compute_gradient(v0, h0, vk, hk);
            
            % update with momentum
            %fprintf('-- momentum update...\n');
            if count > 15
                %use different momentum for starting and final phase of
                %learning
                deltaW = alpha_end * deltaW + (1-alpha_end) * Wgrad;
                deltaa = alpha_end * deltaa + (1-alpha_end) * agrad;
                deltab = alpha_end * deltab + (1-alpha_end) * bgrad;
            else
                deltaW = alpha * deltaW + (1-alpha) * Wgrad;
                deltaa = alpha * deltaa + (1-alpha) * agrad;
                deltab = alpha * deltab + (1-alpha) * bgrad;
            end
            
            % weights and bias update 
            %fprintf('-- weights update...\n');
            Ws = Ws + eta * deltaW;
            a = a + eta * deltaa;
            b = b + eta * deltab;

            % weight decay step
            Ws = Ws - lambda * Ws;
            
            % error computation - L2
            %fprintf('-- compute error...\n');
            error = error + norm(minibatch' - vk);
        end
        
        % compute average error over samples on current epoch
        errors(end + 1) = error / Nd; 

        fprintf('- Epoch %d, Error: %f\n', count, errors(end));
        if count > max_epochs
            fprintf('- Finish!\n');
            % stop training
            break
        end
        
        count = count + 1;
    end
end

