function res = tailleZone(imgClasse,i,j)

        [nlignes,ncolonnes,~] = size(imgClasse);
        
        H=zeros(nlignes,ncolonnes);
        classeCourante = imgClasse(i,j);
        H(imgClasse==classeCourante) = 1;
        [conn,num]=bwlabel(H,8);
        res = length( find(conn==conn(i,j)) );