close all;
clear all;

Ki = 12; %nbr superpixel en y
Kj = 12; %nbr superpixel en x

pourcentageFusion=0.01;

ratio=0.6;%Compression de l'image

n = 9; %Taille voisinage (impaire)
m = 3; %coef pour la distance du kmeans(5) importance de distance spatiale
seuil = 130; %Binarisation

imga = imread('images/viff.000.ppm');
img = imresize(imga, ratio);

%Conversion Lab
C = makecform('srgb2lab');
lab = applycform(img,C);

labd = lab2double(lab);


L(:,:) = labd(:,:,1);
A(:,:) = labd(:,:,2);
B(:,:) = labd(:,:,3);

K=Ki*Kj;

figure('Name','Img Originale')
imagesc(img);

figure('Name','Img en Lab et centres homogènes')
imagesc(lab);

[nlignes,ncolonnes,ncanaux] = size(img);
N = nlignes*ncolonnes;
tailleMoyenne = N/K;

%CALCUL DES CENTRES

%Initialisation des centres
S=sqrt(N/(Ki*Kj));
Si = nlignes/Ki;
Sj = ncolonnes/Kj;
centres = zeros(Ki*Kj,5);
[I,J] = meshgrid(1:ncolonnes,1:nlignes);
l = 1;
for i=1:Ki
    y = min((2*i -1)*round(nlignes/(2*Ki)),nlignes);
    for j=1:Kj
        x = min((2*j -1)*round(ncolonnes/(2*Kj)),ncolonnes);
        centres(l,5) = x;centres(l,4)=y;centres(l,1) = L(y,x);centres(l,2)= A(y,x);centres(l,3)=B(y,x);
        l=l+1;
    end;
end;

%Affichage centres initiaux
hold on;
for p=1:Ki*Kj
    plot(centres(p,5),centres(p,4),'+','MarkerSize',10,'MarkerEdgeColor','b');
end;
%pause;

%Decalage centres
[Fx,Fy] = gradient(L(:,:));
voisinageFx=zeros(n,n);
voisinageFy=zeros(n,n);

for i=1:K
    xcentre = centres(i,5);
    ycentre = centres(i,4);
    
%     minx = max(floor(xcentre - (n-1)/2),1);
%     maxx = min(floor(xcentre + (n-1)/2),ncolonnes);
%     miny = max(floor(ycentre - (n-1)/2),1);
%     maxy = min(floor(ycentre + (n-1)/2),nlignes);
    
    minx = floor(xcentre - (n-1)/2);
    maxx = floor(xcentre + (n-1)/2);
    miny = floor(ycentre - (n-1)/2);
    maxy = floor(ycentre + (n-1)/2);
    if(minx>1 & maxx<ncolonnes & miny>1 & maxy<nlignes)
    
        voisinageFx(1:n,1:n) = Fx(miny:maxy,minx:maxx);
        voisinageFy(1:n,1:n) = Fy(miny:maxy,minx:maxx);
        normvoisi(1:n,1:n) = voisinageFx.^2 + voisinageFy.^2;
    
        [M,Indmin] = min(normvoisi(:));
        [I_row, I_col] = ind2sub(size(normvoisi),Indmin);
    
        di = I_row - (n-1)/2;
        dj = I_col - (n-1)/2;
        xcentre = xcentre+dj;
        ycentre = ycentre+di;
        centres(i,:) = [L(ycentre,xcentre) A(ycentre,xcentre) B(ycentre,xcentre) ycentre xcentre];
    end;
end;

%Affichage des centres deplaces
figure('Name','Déplacement des centres')
imagesc(img);

hold on;
for p=1:K
    plot(centres(p,5),centres(p,4),'+','MarkerSize',10,'MarkerEdgeColor','b');
end;
%pause;


%CLASSIFICATION KMEANS

X = [ L(:) A(:) B(:) I(:) J(:)];
[idx,centres] = kmeans2(X,Ki*Kj,m,S,'start',centres,'distance','sqEuclidean');

%Affichage resultat kmeans
figure('Name','Classification kmeans avec distance modifiée')
imgidx = reshape(idx,nlignes,ncolonnes);
imagesc(imgidx);
%pause;



%FUSION DES REGIONS

imgClasse(:,:) = reshape(idx,nlignes,ncolonnes);
imgClasseNum = zeros(nlignes,ncolonnes);
dernierId = 0;

index = [];

%Etiquettage en zone connexe
for classeCourante=1:K
    H=zeros(nlignes,ncolonnes);
    H(imgClasse==classeCourante)=1;
    [conn,num] = bwlabel(H,8);
    conn(conn~=0) = conn(conn~=0) + dernierId;
    imgClasseNum = imgClasseNum + conn;
    index(dernierId+1:dernierId+num) = classeCourante;
    dernierId = dernierId+num;
end;

conn=imgClasseNum;
num = dernierId;

seuilFusion = floor(pourcentageFusion*tailleMoyenne);
termine = false;
deci=[-1 -1 -1 0 0 +1 +1 +1];%deplacement pour le parcours des pixels adjacents en 8 connexite
decj=[-1 0 1 -1 +1 -1 0 +1];%deplacement pour le parcours des pixels adjacents en 8 connexite

while (~termine)
    termine = true;
    for composanteCourante=1:num
        %progression: composanteCourante/num
        
        taille = length(find(conn==composanteCourante));
        if( taille<=seuilFusion & taille>0)
            %Recherche de la zone voisine la plus grande
            
            %construction vecteur adjacence
            adja = zeros(num,1);
            [r,c,v] = find(conn==composanteCourante);
            for u=1:length(r)
                %Parcours des pixels de la zone connexe
                indi=r(u);
                indj=c(u);
                for g=1:8
                    %Parcours des pixels adjacents à (indi,indj)
                    iadja=indi+deci(g);jadja=indj+decj(g);
                    if(iadja<nlignes & iadja>0 & jadja<ncolonnes & jadja>0)
                        connij = conn(iadja,jadja);
                        if(connij~=composanteCourante)
                            adja(connij) = length(find(conn==connij));
                        end;
                    end;
                end;
            end;
            
            [val,indmax] = max(adja);%choix de la zone adjacente la plus grande
            %Fusion
            conn(conn==composanteCourante) = indmax;
            termine = false;
        end;
    end;
end;

%Réindexage
for i=1:nlignes
    for j=1:ncolonnes
        conn(i,j) = index(conn(i,j));
    end;
end;

figure('Name','Renforcement Connexité');
%affichage_resultat(X,conn(:),K,ncanaux,'Segmentation SLIC',1);
imagesc(conn);


%SEGMENTATION BINAIRE

%centresInterieur = find(centresrgb(:,1)>seuil);
% centresInterieur = find(centresrgb(:,3)>centresrgb(:,1) & centresrgb(:,2)> centresrgb(:,1));
% binarisation = zeros(nlignes,ncolonnes);
% for i=1:length(centresInterieur)
%     H = conn == centresInterieur(i);
%     binarisation = binarisation + H;
% end;
% 
% figure;
% imagesc(binarisation);

C2 = makecform('lab2srgb');
centresrgb = centres;

temp = conn(:);
[i,j] = size(temp);
rres=temp;
gres=temp;
bres=temp;
for i =1:K
    ind=find(conn==i);
    rres(ind)=centresrgb(i,1);
    gres(ind)=centresrgb(i,2);
    bres(ind)=centresrgb(i,3);
end;

rres = reshape(rres,nlignes,ncolonnes);
gres = reshape(gres,nlignes,ncolonnes);
bres = reshape(bres,nlignes,ncolonnes);

res(:,:,1) = rres;
res(:,:,2) = gres;
res(:,:,3) = bres;

res = lab2uint8(res);
res = applycform(res,C2);

figure;
imagesc(res);

figure;
imagesc(res(:,:,1)<seuil);

% centresInterieur = find(centres(:,1)>seuil);
% binarisation = zeros(nlignes,ncolonnes);
% for i=1:length(centresInterieur)
%     H = conn == centresInterieur(i);
%     binarisation = binarisation + H;
% end;
% 
% figure('Name','Binarisation')
% imagesc(binarisation);
