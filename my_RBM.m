%% LOAD DATASET
% Starting from the given data files, in .csv format, import the training
% and test data. The training set is composed of 60000 samples,
% the test set of 10000 samples. The first column of the loaded matrix
% contains the label, the other 784 represent the 28 by 28 grid of pixels
% for the hand written digit.

%train_set = load('mnist_train.csv');
%test_set  = load('mnist_test.csv');

train_labels = train_set(:, 1);
test_labels  = test_set(:, 1);

train_digits = train_set(:, 2:end)/255;
test_digits = test_set(:, 2:end)/255;

Ntrain = size(train_digits, 1); % number of training samples
Ni = size(train_digits, 2);     % number of pixels per training sample 
                                % --> number of inputs 
                                % --> number of visible units

Ntest = size(test_digits, 1);   % number of test samples

%% SHOW ONE SAMPLE DIGIT 
% figure
% colormap gray;
% imagesc(reshape(train_digits(1,:), 28, 28)')
% title(sprintf('Label = %i', train_labels(1,1)));

%% ONE HOT ENCODING
% This step is fundamental to convert the labels into a binary form the
% labels. In this case we have 10 classes (digits labelled from 0 to 9), so
% we convert a label (= a digit) into a 10-bit-long array composed of zeros
% and ones, in which the position of the 1 represent the digit (Zero-based
% one-hot encoding)

encoded_lab_tr   = zeros(size(train_labels, 1), 10);
encoded_lab_test = zeros(size(test_labels, 1), 10);

for i = 1:size(train_labels, 1)
        encoded_lab_tr(i, train_labels(i)+1) = 1;
end

for i = 1:size(test_labels, 1)
        encoded_lab_test(i, test_labels(i)+1) = 1;
end

%% INITIALIZE RBM
% We set the number of hidden units to 100. Then we initialise the biases
% for hidden and visible units to 0. For the weights, we choose them
% sampled at random from a gaussian with zero-mean and small standard
% deviation of 0.01

Nhidden = 196;   % number of hidden units: chosen as 784/4, group pixels by 4

a = zeros(Ni, 1);      % visible units biases
b = zeros(Nhidden, 1); % hidden units biases

Ws = normrnd(0., 0.01, Ni, Nhidden); % weight matrix (size = Ni x Nhidden)

%% TRAINING RBM
% The main training parameters are initialised and the training loop is
% performed. 

max_epochs = 500; % number of training epochs

%eta = 0.001; % learning rate
%eta = 0.0001; % learning rate
eta = 0.01; % learning rate

alpha = 0.5;     % initial momentum
alpha_end = 0.9; % final momentum

lambda = 5e-4;   % regularization 

batch_size = 600;

k = 1; % contrastive-divergence steps

%training process 
t_start = tic;
[Ws, a, b, errors] = training(train_digits, Ws, a, b, k, eta, alpha, alpha_end, lambda, batch_size, max_epochs);

%time taken for training RBM
t_end = toc(t_start);

% save trained network parameters
%save('weights.mat', 'Ws')
%save('bias_visible.mat', 'a')
%save('bias_hidden.mat', 'b')

save('weights_final.mat', 'Ws')
save('bias_visible_final.mat', 'a')
save('bias_hidden_final.mat', 'b')

%% LOAD PREVIOUSLY TRAINED NETWORK
% load trained network parameters
% 
% weigths = matfile('weights_final.mat');
% bias_v  = matfile('bias_visible_final.mat');
% bias_h  = matfile('bias_hidden_final.mat');
% 
% Ws = weigths.Ws;
% a  = bias_v.a;
% b  = bias_h.b;

%% ENCODE DIGITS

%define sigmoid function with function handle
sigmoid = @(a) 1.0 ./ (1.0 + exp(-a));

encoded_train = zeros(Ntrain, Nhidden);
encoded_test  = zeros(Ntest, Nhidden);

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

%% TRAIN SOFTMAX LAYER
% For finally classifying the output
softmax = trainSoftmaxLayer(encoded_train', encoded_lab_tr', 'MaxEpochs', 1000);

% Final predictions!
y_pred_train = softmax(encoded_train');
y_pred_test  = softmax(encoded_test');

%% PLOT ERROR
% To check training performances

figure
x0=10;
y0=10;
set(gcf,'position',[x0,y0])
plot(1:size(errors, 2), errors, 'r', 'LineWidth', 2);
title(sprintf('\\textbf{Training error}, Nh: %i, $\\eta$: %.4f, $\\lambda$: %.1e', Nhidden, eta, lambda), 'interpreter', 'latex');
xlabel(sprintf('\\textbf{Epoch}'), 'FontSize', 12, 'interpreter', 'latex');
ylabel(sprintf('\\textbf{Error}'), 'FontSize', 12, 'interpreter', 'latex');
xlim([0 502]);

matlab2tikz('plot_error.tex', 'width', '3in', 'height', '2.2in'); 

%% PLOT CONFUSION MATRICES
% To compare predictions with real labels

figure
plotconfusion(encoded_lab_tr', y_pred_train);
title(sprintf('\\textbf{Confusion matrix - Training}, Nh: %i, $\\eta$: %.4f, $\\lambda$: %.1e', Nhidden, eta, lambda), 'interpreter', 'latex');
xlabel(sprintf('\\textbf{True labels}'), 'FontSize', 10, 'interpreter', 'latex');
ylabel(sprintf('\\textbf{Predicted labels}'), 'FontSize', 10, 'interpreter', 'latex');
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});
cleanfigure;

matlab2tikz('confusion_train.tex', 'width', '3in', 'height', '3in'); 


figure
plotconfusion(encoded_lab_test', y_pred_test);
title(sprintf('\\textbf{Confusion matrix - Test}, Nh: %i, $\\eta$: %.4f, $\\lambda$: %.1e', Nhidden, eta, lambda), 'interpreter', 'latex');
xlabel(sprintf('\\textbf{True labels}'), 'FontSize', 12, 'interpreter', 'latex');
ylabel(sprintf('\\textbf{Predicted labels}'), 'FontSize', 12, 'interpreter', 'latex');
xticklabels({'0','1','2','3','4','5','6','7','8','9'});
yticklabels({'0','1','2','3','4','5','6','7','8','9'});
cleanfigure;

matlab2tikz('confusion_test.tex', 'width', '3in', 'height', '3in'); 

%% PLOT WEIGHTS 
% to check for unused hidden units and/or to see how the RBM learned

figure
sgtitle(sprintf('\\textbf{Weights visualization} - Nh: %i', Nhidden), 'FontWeight','bold', 'interpreter', 'latex');
hold on

for i=1:Nhidden
   subplot(14, 14, i);
   imshow(reshape(Ws(:,i), 28, 28));
end
cleanfigure;

matlab2tikz('weights196.tex', 'width', '6in', 'height', '6in'); 