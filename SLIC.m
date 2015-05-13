close all;
clear all;

Ki = 10; %nbr superpixel en 
Kj = 10;

pourcentageFusion=0.5;
ratio=0.5;

n = 3; %Taille voisinage 
m = 10;
seuil = 10;

imga = imread('images/viff.000.ppm');
img = imresize(imga, ratio);

K=Ki*Kj;
R(:,:) = img(:,:,1);
V(:,:) = img(:,:,2);
B(:,:) = img(:,:,3);

affichage_image(img,'SLIC',1);
[nlignes,ncolonnes,ncanaux] = size(img);
N = nlignes*ncolonnes;

%Initialisation
S=sqrt(N/(Ki*Kj));
Si = nlignes/Ki;
Sj = ncolonnes/Kj;
centres = zeros(Ki*Kj,5);
[I,J] = meshgrid(1:ncolonnes,1:nlignes);
l = 1;
for i=1:Ki
    y = (2*i -1)*round(nlignes/(2*Ki));
    for j=1:Kj
        x = (2*j -1)*round(ncolonnes/(2*Kj));
        %[x y R(y,x) V(y,x) B(y,x)]
        %centres(l,:) = [x y R(y,x) V(y,x) B(y,x)];
        centres(l,5) = x;centres(l,4)=y;centres(l,1) = R(y,x);centres(l,2)= V(y,x);centres(l,3)=B(y,x);
        l=l+1;
    end;
end;

hold on;
for p=1:Ki*Kj
    plot(centres(p,5),centres(p,4),'+','MarkerSize',10,'MarkerEdgeColor','b');
end;
pause;

% %Decalage centres
% [Fx,Fy] = gradient(img(:,:,));
% for i=1:K*K
%     xcentre = centres(i,1);
%     ycentre = centres(i,2);
%     
%     voisinageFx = Fx(floor(xcentre - (n-1)/2):floor(xcentre + (n-1)/2),floor(ycentre - (n-1)/2):floor(ycentre + (n-1)/2));
%     voisinageFy = Fy(floor(xcentre - (n-1)/2):floor(xcentre + (n-1)/2),floor(ycentre - (n-1)/2):floor(ycentre + (n-1)/2));
%     normvoisi = Fx.^2 + Fy.^2;
%     
%     [M,I] = min(normvoisi(:));
%     [I_row, I_col] = ind2sub(size(A),I);
%     
%     di = I_row - (n-1)/2;
%     dj = I_col - (n-1)/2;
%     xcentre = xcentre+di;
%     centres(i,:) = [xcentre ycentre R(xcentre,ycentre) V(xcentre,ycentre) B(xcentre,ycentre)];
% end;


X = double([ R(:) V(:) B(:) I(:) J(:)]);
% xi = ceil(x);
% yi = ceil(y);
% A = zeros(nb_classes,3);
% for i=1:nb_classes
%     A(i,1) = R(yi(i), xi(i));
%     A(i,2) = V(yi(i), xi(i));
%     A(i,3) = B(yi(i), xi(i));
% end;

% %
% idx = zeros(nlignes*ncolonnes,1);
% E = 100000;
% while (E>=seuil)
%     E=0;
%     for h=1:nlignes*ncolonnes
%        for g=1:Ki*Kj
%            dspa = sqrt((X(h,4) - centres(g,4))^2 + (X(h,5) - centres(g,5))^2);
%            if dspa>S
%                D(h,g) = Inf;
%            else
%                dcouleur = sqrt((X(h,1) - centres(g,1)).^2 + (X(h,2) - centres(g,2)).^2 + (X(h,3) - centres(g,3)).^2);
%                D(h,g) = dcouleur + (m/S)*dspa;
%            end;
%         end;
%         
%         %Maj des classes
%         [val,ind] = min(D(h,:));
%         idx(h) = ind;
%     end;
%     
%     
%     
%     for i=1:Ki*Kj
%         ptsclasse = X(find(idx==i),:);
%         mptsclasse = mean(ptsclasse);
%         xa=mptsclasse(1,5);
%         ya=mptsclasse(1,4);
%         Ra=mptsclasse(1,1);
%         Va=mptsclasse(1,2);
%         Ba=mptsclasse(1,3);
%         
%         dE = (centres(i,5)-xa)^2 + (centres(i,4)-ya)^2;
%         E = E + dE;
%         centres(i,5) = xa;centres(i,4)=ya;centres(i,1) = Ra;centres(i,2)= Va;centres(i,3)=Ba;
%     end;
%     E
% end;

[idx,centres] = kmeans2(X,Ki*Kj,m,S,'start',centres,'distance','sqEuclidean');

affichage_resultat(X,idx,Ki*Kj,ncanaux,'Segmentation (NVG pos) en ',1);

%FUSION DES REGIONS

imgClasse(:,:) = reshape(idx,nlignes,ncolonnes);
pourcentageFusion
bwlabel(imgClasse,8);
