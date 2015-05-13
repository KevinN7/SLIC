function affichage_image(I_entree,message,numero)

figure(numero);
figure('Name',message);
imagesc(I_entree);
if size(I_entree,3)==1
	colormap(gray);
end;
axis equal;
axis off;
