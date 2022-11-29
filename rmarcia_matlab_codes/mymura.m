clear

p = 193;
p = 101;
M = zeros(p);

M(2:end, 1) = 1;

C = -ones(p,1);
for i = 1:p-1
  C(mod(i*i, p)) = 1;
end

for i = 1:p-1
for j = 1:p-1
if (C(i)*C(j) == 1)
     M(i+1,j+1) = 1;
end
end
end

G    = 2*M - ones(p);  % decoding pattern
G(1) = 1;

fG = fft2(G);
fM = fft2(M);
MG = real(ifft2(fG .* fM));

MG = circshift(MG, [(p-1)/2  (p-1)/2]);
%MG = MG .* (abs(MG) > 0.00000001);

figure(1);
imshow(uint8(255*M), 'Border', 'tight');
imagesc(uint8(255*M));
colormap(gray); axis off
axis equal
axis image

H = uint8(255*ones(p, p, 3));
H(:,:,1) = uint8(127*(G+1));
H(:,:,2) = uint8(127*(G+1));


figure(2)
imagesc(H);
axis equal; axis off; axis image


figure(3)
mesh(MG)
set(gca,'XGrid', 'on', 'YGrid', 'on');
set(gca,'ZGrid', 'on');
set(gca, 'XLim', [1 p])
set(gca, 'YLim', [1 p]);
set(gca, 'ZLim', [0 max(max(MG))]);
set(gca, 'ZTick', [0: 1000: max(max(MG))]);



