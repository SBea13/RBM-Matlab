function [Wgrad, agrad, bgrad] = compute_gradient(v0, h0, vk, hk)
    Wgrad = (v0 * h0') - (vk * hk');
    agrad = (sum(v0,2) - sum(vk,2));
    bgrad = (sum(h0,2) - sum(hk,2));
end

