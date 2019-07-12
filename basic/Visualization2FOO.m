clear all; clc; close all;

disp('Welcome to the network visualization tool for communicability space.')

% disp('Please, select the network file')

% [FileName,PathName,FilterIndex] = uigetfile('*.txt','Please, select the network file you want to plot');
PathName = '/home/renato/repos/communicabilityLayout/data/'

FileName = 'dolphinsA.txt'
FileName = 'polblogs_A_cc.txt'

warning('off','all')

% rng(2); %Seed for random number generation

fname=fullfile(PathName,FileName);
A=load(fname);

A=max(A,A')-diag(diag(A)); %This symetrizes the adyacency matrix
n=length(A); 
% 
% m=input('Do you want to change the temperature value?, Y/N [N]:','s');

beta=1; %Temperature

% 
% ncluin=str2num(input('Select the minimum number of clusters you want to study:','s'));
% nclu=str2num(input('Select the maximun number of clusters you want to study:','s'));
%
ncluin = 2
nclu = 6 
G=expm(beta*A); %Communicability matrix
sc=diag(G); %Vector of self-communicabilities
u=ones(n,1);
CD= (sc*u'+u*sc'-2*G); %Squared communicability distance matrix
X= CD.^.5; %Communicability distance matrix
