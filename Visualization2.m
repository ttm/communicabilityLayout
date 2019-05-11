clear all; clc; close all;

disp('Welcome to the network visualization tool for communicability space.')

% disp('Please, select the network file')

% [FileName,PathName,FilterIndex] = uigetfile('*.txt','Please, select the network file you want to plot');
PathName = '/home/renato/repos/communicabilityLayout/'
FileName = 'dolphinsA.txt'

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
An__=acosd(G./((sc*u').*(u*sc')).^.5); %Communicability angles matrix
An_=real(An__+1e-5-1e-5*eye(n,n));
An=max(An_,An_');
% 
E_original=eig(An);
% 
dims=3; %dimensions to try
metrics = {'metricstress', 'metricsstress', 'sammon','strain'};
% 
% disp('Please, wait')
% 
for j=1:length(metrics)
    try
        
        [Y{j},stress(j)]=mdscale(An,dims,'Criterion',metrics{j},'Start', 'random','Replicates', 30, 'Options',statset('MaxIter',1000));
        
        % matlab plot
        %figure2=figure;
        %gplot23D(A,Y{j},'-o'), axis square %Plotting the results
        %vector_labels=[1:n]';  %generate array of data points' markers 
        %labels=cellstr(num2str(vector_labels)); %transform the array into a proper format before calling "tex"
 
        %for i=1:n
        %    text(Y{j}(i,1),Y{j}(i,2),Y{j}(i,3),labels(i),'VerticalAlignment','bottom','HorizontalAlignment','right'); %You can customize alignments and other properties see help for text
        %end

        %title([metrics{j},'. Num of dimensions = ',num2str(dims)]);
        %axis square;
        %print([PathName,'\matlabPlot-',metrics{j}],'-dpng')
        
        error(j)=0;

    catch ME
    fprintf('Metric %s cannot be used: %s\n', metrics{j}, ME.message);
    error(j)=1;
    continue;  % Jump to next iteration of: for i
  end
    
end

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
        
csvwrite([PathName,'angles-Gpq.txt'],[enlaces,GPqs])

% Export subgraph centralities


%Recuperar los angulos en la nueva metrica
%atan2(norm(cross(u,v)),dot(u,v)); %la implementacion es mas precisa para angulos peque?os segun la documentacion de matlab

for z=1:length(metrics)
    for i=1:length(Y{z})
        for j=1:length(Y{z})
            newAngles{z}(i,j)=atan2d(norm(cross(Y{z}(i,:),Y{z}(j,:))),dot(Y{z}(i,:),Y{z}(j,:)));
            %newangles2(i,j)=acosd(dot(Y{1}(i,:),Y{1}(j,:))/(norm(Y{1}(i,:))*norm(Y{1}(j,:))));
        end
    end
end


% Distance between metrics and the original representation
for j=1:length(metrics)
    if (error(j)~=1)
        E_metrics{j} = eig(newAngles{j});       % Get eigenvalues 
        %Distances between eigenvalue value vectors
        Metric_distance(j) = sqrt((1/n)*(sum((E_original-E_metrics{j}).^2)));
    else
        Metric_distance(j) = Inf;
    end
    

end

%Best metric
[~,index] = min(Metric_distance);
best=metrics(index);

%Exporting coordinates
csvwrite([PathName,'coords-sc.txt'],[Y{index},sc]);

%Export Angles full matrix
csvwrite([PathName,'anglesFull.txt'],An);




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
