clear all; clc; close all;

disp('Welcome to the network visualization tool for communicability space.')

disp('Please, select the network file')

[FileName,PathName,FilterIndex] = uigetfile('*.txt','Please, select the network file you want to plot');

warning('off','all')


fname=fullfile(PathName,FileName);
A=load(fname);

A=max(A,A')-diag(diag(A)); %This symetrizes the adyacency matrix
n=length(A); 

m=input('Do you want to change the temperature value?, Y/N [N]:','s');
if m=='Y'
    prompt = 'Introduce the new value for the temperature parameter beta? ';
    beta=input(prompt);
else
    beta=1; %Temperature
end

ncluin=str2num(input('Select the minimum number of clusters you want to study:','s'));
nclu=str2num(input('Select the maximun number of clusters you want to study:','s'));

G=expm(beta*A); %Communicability matrix
sc=diag(G); %Vector of self-communicabilities
u=ones(n,1);
CD= (sc*u'+u*sc'-2*G); %Squared communicability distance matrix
X= CD.^.5; %Communicability distance matrix
An=acosd(G./((sc*u').*(u*sc')).^.5); %Communicability angles matrix
An=real(An+1e-5-1e-5*eye(n,n));
An=max(An,An');

E_original=eig(An);

dims=3; %dimensions to try
metrics = {'metricstress', 'metricsstress', 'sammon','strain'};

disp('Please, wait')


% Exporting original angles and Communicabilities
triu_An=triu(An);
triu_An_con_zeros=triu_An.*A;
triu_An_con_zeros=triu_An_con_zeros';
enlaces_con_zeros=triu_An_con_zeros(:)';
enlaces = enlaces_con_zeros(enlaces_con_zeros~=0);
enlaces=enlaces';

triu_G=triu(G);
triu_G_con_zeros=triu_G.*A;
triu_G_con_zeros=triu_G_con_zeros';
Gpq_con_zeros=triu_G_con_zeros(:)';
GPqs = Gpq_con_zeros(Gpq_con_zeros~=0);
GPqs=GPqs';


% Select the best number of clusters with CalinskiHarabasz criterion
eva = evalclusters(An,'kmeans','Silhouette','KList',[ncluin:nclu]);

fprintf('The optimal number of clusters is: %d \n',eva.OptimalK);

figure1=figure;
values(ncluin:nclu)=eva.CriterionValues;
plot(values)
xlim([ncluin nclu])
xlabel('Number of clusters')
ylabel('Silhouette criterion')

print([PathName,'\Silhouette'],'-dpdf')

% Train kmeans clustering
clusters=kmeans(An, eva.OptimalK);


csvwrite([PathName,'clusters.txt'],clusters);


for j=1:max(clusters)
modules{j}=find(clusters==j);
end
modularity=modularity_metric(modules,A);


warning('on','all')
disp('Completed succesfully')
