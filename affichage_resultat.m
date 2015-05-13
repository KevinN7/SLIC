function affichage_resultat(X,idx,nb_classes,nb_canaux,message,numero)

% Affectation de couleurs aux différentes classes :
couleurs = lines(nb_classes);

figure(numero);
hold on;
for k = 1:nb_classes
	plot(X(idx==k,nb_canaux+1),-X(idx==k,nb_canaux+2),'.','Color',couleurs(k,:));
end
axis equal;
axis off;
