%% Nuri Mert Vural- 201801286
%% Music Sequence
clc;
close all;
clear;
%Load data
load('nngc_mus.mat')
epoch=1500;

in_size=size(nngc_data(:,1:end-1),1);


%% SRNN - 8000
close all;
s_rnn=SRNN(ones(in_size,1)',movmean(nngc_data(:,end)',1),75);
s_rnn.SGD_train(5e-5,0.95,epoch)
%% CWRNN - 8000
close all;
cw_rnn=CWRNN(zeros(in_size,1)',movmean(nngc_data(:,end)',25),14,9);
cw_rnn.SGD_train(1e-4,0.95,epoch)
%% LSTM- 8000.
close all
lstm1=LSTM(zeros(in_size,1)',movmean(nngc_data(:,end)',25),42);
lstm1.SGD_train(3e-5,0.95,3500);

%% Statistical test
x_input=[zeros(in_size,1)' ; ones(in_size,1)'];

lstm1.evaluate(x_input)
lstm_seq=lstm1.final_outputs;

cw_rnn.soft_reset()
cw_rnn.evaluate(x_input)
cw_seq=cw_rnn.final_outputs;

%%
figure(2); hold on;
plot(lstm_seq,'LineWidth',1.5); grid;
plot(cw_seq,'LineWidth',1.5);
plot(1:length(cw_seq),movmean(nngc_data(:,end)',25),'LineWidth',1.5);
[h,prob]=ttest(lstm_seq-cw_seq)
var_lstm=var(lstm_seq-movmean(nngc_data(:,end)',25))
var_cw=var(cw_seq-movmean(nngc_data(:,end)',25))
