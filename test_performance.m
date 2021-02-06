% TESTING TRAINING PERFORMANCES
% Given a set of metaparameters for the RBM, this program test the training
% performance, after having loaded all the data

Nhidden = [ 100; 196; 16];   % number of hidden units
eta = [1e-2; 1e-3; 1e-4];    % learning rate
lambda = [1e-4; 5e-4; 1e-5]; % weight decay
times  = [];

%define sigmoid function with function handle
sigmoid = @(a) 1.0 ./ (1.0 + exp(-a));

for nh=1:3
    for l=1:3
        for e=1:3
            filename_w = 'Nh'+string(Nhidden(nh))+'lambda'+string(lambda(l))+'eta'+string(eta(e))+'Ws.mat'
            filename_a = 'Nh'+string(Nhidden(nh))+'lambda'+string(lambda(l))+'eta'+string(eta(e))+'a.mat'
            filename_b = 'Nh'+string(Nhidden(nh))+'lambda'+string(lambda(l))+'eta'+string(eta(e))+'b.mat'
            
            % INITIALIZE RBM

            a = zeros(Ni, 1);      % visible units biases
            b = zeros(Nhidden(nh), 1); % hidden units biases

            Ws = normrnd(0., 0.01, Ni, Nhidden(nh)); % weight matrix (size = Ni x Nhidden)

            %TRAINING RBM
             
            max_epochs = 100; % number of training epochs

            alpha = 0.5;     % initial momentum
            alpha_end = 0.9; % final momentum

            batch_size = 600;

            k = 1; % contrastive-divergence steps
            
            t_start = tic;
            %training process
            [Ws, a, b, errors] = training(train_digits, Ws, a, b, k, eta(e), ...
                                          alpha, alpha_end, lambda(l), ....
                                          batch_size, max_epochs);
            
            %time taken for training RBM
            times = [times, toc(t_start)];
            
            % save trained network parameters
            save(filename_w, 'Ws')
            save(filename_a, 'a')
            save(filename_b, 'b')
            
            % ENCODE DIGITS

            encoded_train = zeros(Ntrain, Nhidden(nh));
            encoded_test  = zeros(Ntest, Nhidden(nh));

            % encode training digits
            for i = 1:Ntrain
                v0 = train_digits(i, :)'; 
                h0 = sigmoid(Ws' * v0 + b); 

                encoded_train(i, :) = h0;
            end

            % encode test digits
            for i = 1:Ntest
                v0 = test_digits(i, :)'; 
                h0 = sigmoid(Ws' * v0 + b); 

                encoded_test(i, :) = h0;
            end

            % TRAIN SOFTMAX LAYER
            % For finally classifying the output
            softmax = trainSoftmaxLayer(encoded_train', encoded_lab_tr', 'MaxEpochs', 1000);

            % Final predictions!
            y_pred_train = softmax(encoded_train');
            y_pred_test  = softmax(encoded_test');

            % PLOT ERROR
            % To check training performances

            figure
            plot(1:size(errors, 2), errors, 'r', 'LineWidth',2 );
            title(sprintf('\\textbf{Training error}, Nh: %i, $\\eta$: %.4f, $\\lambda$: %.1e',...
                            Nhidden(nh), eta(e), lambda(l)), 'interpreter', 'latex');
            xlabel(sprintf('\\textbf{Epoch}'), 'FontSize', 12, 'interpreter', 'latex' );
            ylabel(sprintf('\\textbf{Error}'), 'FontSize', 12, 'interpreter', 'latex');
            xlim([0 105]);

            % PLOT CONFUSION MATRICES
            % To compare predictions with real labels

            figure
            plotconfusion(encoded_lab_tr', y_pred_train);
            title(sprintf('\\textbf{Confusion matrix - Training}, Nh: %i, $\\eta$: %.4f, $\\lambda$: %.1e', ...
                            Nhidden(nh), eta(e), lambda(l)), 'interpreter', 'latex');
            xlabel('True digits', 'FontWeight','bold');
            ylabel('Predicted digits', 'FontWeight','bold');
            xticklabels({'0','1','2','3','4','5','6','7','8','9'});
            yticklabels({'0','1','2','3','4','5','6','7','8','9'});

            figure
            plotconfusion(encoded_lab_test', y_pred_test);
            title(sprintf('\\textbf{Confusion matrix - Test},  Nh: %i, $\\eta$: %.4f, $\\lambda$: %.1e', ...
                            Nhidden(nh), eta(e), lambda(l)), 'interpreter', 'latex');
            xlabel('True digits', 'FontWeight','bold');
            ylabel('Predicted digits', 'FontWeight','bold');
            xticklabels({'0','1','2','3','4','5','6','7','8','9'});
            yticklabels({'0','1','2','3','4','5','6','7','8','9'});


            % PLOT WEIGHTS 
            % to check for unused hidden units and/or to see how the RBM learned

            figure
            sgtitle(sprintf('Weights visualization - Nh: %i', Nhidden(nh)), 'FontWeight','bold');
            hold on

            for i=1:Nhidden(nh)
               subplot(sqrt(Nhidden(nh)), sqrt(Nhidden(nh)), i);
               imshow(reshape(Ws(:,i), 28, 28));
            end
        end
    end
end



