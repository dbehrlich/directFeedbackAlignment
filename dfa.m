%Non-linearities

logis =@(x) 1./(1+exp(-x));
dlogis =@(x) logis(x).*(1-logis(x));

%Input Output
in = [1,1 ; 0,1 ; 1,0 ; 0,0]';
out = [0,1,1,0];
y = zeros(size(out));

n_hidden = 10; % Number of hidden units
num_iterations = 1000; %Number of learning steps
num_traces = 4000;

for jj = 1:num_traces
    
%Random initialization of weight matrix
w1 = randn(size(in,1),n_hidden)';
w2 = randn(n_hidden,n_hidden)';
w3 = randn(n_hidden,size(out,1))';


%Copy of same initialization for Backprop
bp_w1 = w1;
bp_w2 = w2;
bp_w3 = w3;


%Random Error Feedback matrices
B1 = rand(n_hidden,1);
B2 = randn(n_hidden,1);




    for ii = 1:num_iterations


        %Direct Feedback Alignment

        %Forward Pass 
        a1 = w1*in;
        z1 = logis(a1);

        a2 = w2*z1;
        z2 = logis(a2);

        ay = w3*z2;
        y = logis(ay);

        e = y-out;

        %Feedback 
        d_a1 = (B1*e).*dlogis(a1);
        d_a2 = (B2*e).*dlogis(a2);

        %Weight Updates
        dw1 = -d_a1*in';
        dw2 = -d_a2*z1';
        dw3 = -e*z2';

        w1 = w1 + dw1;
        w2 = w2 + dw2;
        w3 = w3 + dw3;

        %Backprop

        %Forward Pass
        bp_a1 = bp_w1*in;
        bp_z1 = logis(bp_a1);

        bp_a2 = bp_w2*bp_z1;
        bp_z2 = logis(bp_a2);

        bp_ay = bp_w3*bp_z2;
        bp_y = logis(bp_ay);

        bp_e = bp_y-out;

        %Weight Updates
        bp_d3 = (bp_y-out).*dlogis(bp_ay);
        bp_d2 = (bp_w3'*bp_d3).*dlogis(bp_a2);
        bp_d1 = (bp_w2'*bp_d2).*dlogis(bp_a1);

        bp_w1 = bp_w1 - (in*bp_d1')';
        bp_w2 = bp_w2 - (z1*bp_d2')';
        bp_w3 = bp_w3 - (z2*bp_d3')';


        if jj == 1
            y_store(:,ii) = y;
        end
        
        e_store(ii,jj) = sum(abs(e));

        bp_e_store(ii,jj) = sum(abs(bp_e));



    end
end

subplot(4,2,1); 
plot(y_store(1,:)'); title('1,1')
subplot(4,2,2);
plot(y_store(2,:)','r'); title('0,1')
subplot(4,2,3); 
plot(y_store(3,:)','g'); title('1,0')
subplot(4,2,4);
plot(y_store(4,:)','c'); title('0,0')

subplot(2,1,2); 
plot([e_store(:,1),bp_e_store(:,1)]); 
legend('DFA','Backprop')
title('error')
xlabel('Training Iterations')

figure;
plot([mean(e_store,2),mean(bp_e_store,2)]); 
legend('DFA','Backprop')
title('Average Error (4000 traces)')
xlabel('Training Iterations')

