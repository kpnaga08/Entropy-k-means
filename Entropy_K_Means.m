clear all;close all;clc

%% Entropy K-Means Clustering with Feature Reduction Under Unknown Number of Clusters 
%% Written by Kristina Pestaria Sinaga (kristinasinaga57@yahoo.co.id)

%% Experiment 1: Numerical data sets

  % Objectives to simultaneously find the number of cluster with feature
  % feature-reduction step without a given number of clusters


%% A two manifold clusters 

% load manifo2.mat % A 3-D dataset 1


%% A three spherical clusters

load data4.mat % A 3-D dataset 2

%% IRIS Data

load fisheriris.mat ;
points    = meas;
[c,ia,ib] = unique(species) ;
label     = ib ;  



points_n   = size(points,1);
points_dim = size(points,2);


%% Initialize Feature Weight
w = ones(1,points_dim)/points_dim;

%% Entropy-regularized k-means 

thres = 0.01;
r1    = 1;
beta  = 1;
rate  = 0;

cluster_n = points_n;
clust_cen = points;
alpha     = ones(1,cluster_n)*1/cluster_n;

err   = 10;
err_w = 10;
index_wf_red_all = [];

while and (err>thres,cluster_n>1)

    tic
    rate=rate+1;
    
    %% STEP 1 : Compute Delta
    
    delta=abs(mean(points))./var(points);
    Gamma=exp(-cluster_n/300);
    
    
    %% STEP 2 : Compute Membership
    
    u=zeros(points_n,cluster_n);
    D8=[];
    for k=1:cluster_n
        D1=bsxfun(@minus,points,clust_cen(k,:));
        D2=D1.^2;
        D3=delta.*w;
        D4=bsxfun(@times,D2,D3);
        D5=sum(D4,2);
        D6=Gamma*log(alpha(k));
        D7=bsxfun(@minus,D5,D6);
        D8=[D8 D7];
    end
        
    if rate==1 
        D7(D7==0)=NaN;
        D9=D8;
        D8(logical(eye(size(D8))))=NaN;
        [val idx]=min(D8,[],2);
        D8(isnan(D8))=diag(D9);
    else
        [val idx]=min(D8,[],2);
    end    
    

    
    for i=1:points_n
        u(i,idx(i))=1;
    end
    
    
    %% STEP 3 : Update Feature-weights
    
    if size(w,2)>1
        W4=[]; 
        for k=1:cluster_n
            W1=bsxfun(@minus,points,clust_cen(k,:));
            W2=W1.^2;
            W3=bsxfun(@times,W2,u(:,k));
            W4=[W4;sum(W3,1)];
        end
        W5=sum(W4,1);
        W5(isnan(W5))=0;
        W6=bsxfun(@times,W5,delta);
        W7=(-1*W6)./points_n;
        W8=exp(W7);
        W9=1./delta;
        W10=bsxfun(@times,W8,W9);
        W11=sum(W10,2);
        new_w=bsxfun(@rdivide,W10,W11);
    end
    new_w;
    
    
    %% STEP 4 : Feature reduction proccessing
   
    % DISCARD WEIGHT
    index_wf_red=[];
    thres_reduce=1/sqrt(points_n*points_dim*cluster_n);
    
    index_w=find(new_w<thres_reduce);
    index_wf_red=[index_wf_red index_w];
    index_w_ok=find(new_w>=thres_reduce);
       
    
    
    %% STEP 5: ADJUST WEIGHT
    
    adj_w=new_w;
    adj_w(index_w)=[];
    adj_w=adj_w/sum(adj_w);
    new_w=adj_w;
    
    % Update the number of dimension
    
    new_points_dim=size(new_w,2);
    index_wf_red_all=[index_wf_red_all index_wf_red];

    % Adjust points
    
    new_points=points;
    new_points(:,index_w)=[]; 
    
    % Adjust the cluster centers
    
    clust_cen(:,index_w)=[];
    
    %% STEP 6 : Update the mixing proportion alpha
    new_alpha=sum(u,1)/points_n+beta/Gamma*alpha.*(log(alpha)-sum(alpha.*log(alpha)));
    
    eta=min(1,(1/rate^floor(rate/2-1)));
    
    %% STEP 7 : Compute Beta
    
    temp9=0; 
    for k=1:cluster_n
        temp8=exp(-eta*points_n*abs(new_alpha(k)-alpha(k)));
        temp9=temp9+temp8;
    end
    temp9=temp9/cluster_n;
    temp10=1-max(sum(u,1)/points_n);
    temp11=sum(alpha.*log(alpha));
    temp12=temp10/(-max(alpha)*temp11);

    new_beta=min(temp9,temp12);    
    
    %% STEP 8 : Update the number of clusters c
    
    index=find(new_alpha<=1/((points_n-1)*points_n));
    
    % Adjust alpha
    
    adj_alpha=new_alpha;
    adj_alpha(index)=[];
    adj_alpha=adj_alpha/sum(adj_alpha);
    new_alpha=adj_alpha;
    if size(new_alpha,2)==1
        new_alpha=alpha;
        break;
    end
    
    % Update the number of clusters
    
    new_cluster_n=size(new_alpha,2);
    
    % Adjust memberships
    
    adj_u=u;
    adj_u(:,index)=[];
    adj_u=bsxfun(@rdivide,adj_u,sum(adj_u,2));
    adj_u(isnan(adj_u))=0;

    new_u=adj_u;
    
    if and(rate>=60,new_cluster_n-cluster_n==0)
        new_beta=0;
    end

    
    %% STEP 9 : Update the cluster centers
    
    new_clust_cen=[];
    for k=1:new_cluster_n
            temp4=zeros(1,new_points_dim);
            temp5=0;
            for i=1:points_n
                temp4=temp4+new_u(i,k)*new_points(i,:);
                temp5=temp5+new_u(i,k);
            end
        new_clust_cen=[new_clust_cen; temp4/temp5];
        
    end
    new_clust_cen;
%     nandata2=[new_clust_cen];
%     xdata2=(1:size(nandata2,1))';
%     new_clust_cen=bsxfun(@(x,y) interp1(y(~isnan(x)),x(~isnan(x)),y),nandata2,xdata2);

    
        
    
    error=[];
    for k=1:new_cluster_n    
        error=[error;norm(new_clust_cen(k,:)-clust_cen(k,:))];
    end    
    err=max(error);
    
    
    u=new_u;
    w=new_w;
    points=new_points;
    points_dim=new_points_dim;
    alpha=new_alpha;
    beta=new_beta;
    clust_cen=new_clust_cen;
    cluster_n=new_cluster_n;
    
    
   
end
cluster_n

clust=[];
for i=1:points_n
    [num idx]=max(u(i,:));
    clust=[clust;idx];
end

 AR=1-ErrorRate(label,clust,cluster_n)/points_n
 
 
%% Displaying the output
 
 d=figure;
 
 % Defaults for this blog post
 
 width = 3;     % Width in inches
 height = 3;    % Height in inches
 alw = 0.75;    % AxesLineWidth
 fsz = 12;      % Fontsize
 lw = 0.5;      % LineWidth
 msz = 3;       % MarkerSize
 
 % Here we preserve the size of the image when we save it.
 set(gcf,'InvertHardcopy','on');
 set(gcf,'PaperUnits', 'inches');
 % set(gca,'XTickLabel',[{'0','0.2','0.4','0.6','0.8','1.0'}]);
 % set(gca,'YTickLabel',[{'0','0.2','0.4','0.6','0.8','1.0'}]);
 papersize = get(gcf, 'PaperSize');
 left = (papersize(1)- width)/2;
 bottom = (papersize(2)- height)/2;
 myfiguresize = [left, bottom, width, height];
 set(gcf,'PaperPosition', myfiguresize);

 set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
 set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
 set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
 set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
 
 col=jet(size(unique(clust),1));

 hold on
 
 gscatter(points(:,1),points(:,2),clust,col,'ooooo',7,'off');
 str=sprintf('\nIteration = %d, Number of clusters = %d',rate,cluster_n);
 title(str);
 scatter(clust_cen(:,1),clust_cen(:,2),30,'o','black','filled');
 legend('off');
 
 xlabel('x_{1}');
 ylabel('x_{2}');
 
 for k=1:cluster_n
    idx=(clust==k);

    Mu=mean(points(idx,:));
    X0=bsxfun(@minus,points(idx,:),Mu);

    STD=2;                     %# 2 standard deviations
    conf=2*normcdf(STD)-1;     %# covers around 95% of population
    scale=chi2inv(conf,2);     %# inverse chi-squared with dof=#dimensions

    Cov=cov(X0)*scale;
    [V D]=eig(Cov);
    [D order]=sort(diag(D), 'descend');
    D=diag(D);
    V=V(:, order);
    
    %axis([1.5 4 1.5 4]);

    t=linspace(0,2*pi,100);
    e=[cos(t);sin(t)];        %# unit circle
    VV=V*sqrt(D);             %# scale eigenvectors
    e=bsxfun(@plus,VV*e,Mu'); %#' project circle back to orig space

    plot(e(1,:),e(2,:),'Color','k','Linewidth',1);
 end

 hold off
 
 mean_group=clust_cen;

 
