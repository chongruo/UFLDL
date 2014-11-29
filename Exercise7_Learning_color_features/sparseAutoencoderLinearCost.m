function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    


W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));


% Add one to the end of each input
[data_m,data_n] = size(data);

%% forward
z2 = [W1,b1]*[data;ones(1,data_n)];
a2 = 1./(1+exp(-z2));
z3 = [W2,b2]*[a2;ones(1,data_n)];
a3 = z3;

rou_real = sum(a2,2)./data_n;

% computing the cost
term1 = (1/(2*data_n)) * sum(sum( (a3-data).^2 )); 
term2 = sum(sum(W1.^2)) + sum(sum(W2.^2));
term3 = kl(sparsityParam,rou_real);
cost = term1 + (lambda/2)*term2 + beta*term3;

%% BackPropagation
% for i = 1:data_n                        % traverse all training data
%     cita3 = -(data(:,i)-a3(:,i)).* (a3(:,i)*(1-a3(:,i)));
%     cita2 = a2(:,i).*(1-a2(:,i)).* ((W2'*cita3) + ...
%             beta*( -sparsityParam./rou_real + (1-sparsityParam)./(1-rou_real)) );
%     
%     W1grad = W1grad + cita2 * data(1:end-1,i);
%     W2grad = W2grad + cita3 * a2';
%     b1grad = b1grad + cita2;
%     b2grad = b2grad + cita3;
% end

cita3 = -(data-a3);
cita2 = a2.*(1-a2).*(...
            (W2'*cita3) +... 
             beta*(...
                 repmat((-sparsityParam./rou_real + (1-sparsityParam)./(1-rou_real)),1,data_n))...
            );
W1grad = cita2 * data';
W2grad = cita3 * a2';
b1grad = sum(cita2,2);
b2grad = sum(cita3,2);



W1grad = (1/data_n) * W1grad + lambda* W1;
W2grad = (1/data_n) * W2grad + lambda* W2;
b1grad = (1/data_n) * b1grad;
b2grad = (1/data_n) * b2grad;
    

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end



%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function result = kl(rou,rou_real)
    result = sum( rou.*log(rou./rou_real) + (1-rou).*log((1-rou)./(1-rou_real))   );
end
