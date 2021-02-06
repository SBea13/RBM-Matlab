function [h0, v0, vk, hk] = rbm_CD_k(Ws, a, b, cd_k, x)
    Nh = size(b, 1);
    Ni = size(a, 1);
    batch_size = size(x, 2);
    sigmoid = @(a) 1.0 ./ (1.0 + exp(-a));
    
    b  = repmat(b, 1, batch_size);
    a  = repmat(a, 1, batch_size);
    
    % clamp training vector to visible units
    v0 = x; 
    
    % update hidden units
    p_h0v0 = sigmoid(Ws' * v0 + b);
    h0 = p_h0v0 > rand(Nh, batch_size);
    
    vk = v0;
    hk = h0;
    
    for k = 1:cd_k
        % update visible units to get reconstruction
        p_vkhk = sigmoid(Ws * hk + a);
        vk = p_vkhk; 
        
        % update hidden units again
        p_hkvk = sigmoid(Ws' * vk + b);
        hk = p_hkvk > rand(Nh, batch_size);
        
    end
    
    hk = p_hkvk;
    h0 = p_h0v0;
end

