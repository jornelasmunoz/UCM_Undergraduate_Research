clear

p = 193;
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


G = 2*M - ones(p);
G(1) = 1;

fG = fft2(G);
fM = fft2(M);
MG = real(ifft2(fG .* fM));

MG = circshift(MG, [(p-1)/2  (p-1)/2]);
MG = MG .* (abs(MG) > 0.00000001);

figure(1);
imshow(uint8(255*M), 'Border', 'tight');


a = imread('Smile.jpg', 'JPEG');
a = uint8(rgb2gray(a));

b  = a(5:297, 5:297);
w  = conv2(double(b), ones(2));
x  = w(1:2:end, 1:2:end);

b = zeros(193);
b( 23:169, 23:169) = x;

fb = fft2(double(b));
fM = fft2(M);
Mb = real(ifft2(fb .* fM));
Mb = Mb - min(min(Mb));
Mb = 255*Mb/(max(max(Mb)));

figure(2);
imshow(uint8(Mb), 'Border', 'tight');

fMb = fft2(Mb);
MbG = real(ifft2(fMb .* fG));
MbG = MbG - min(min(MbG));
MbG = 255*MbG/(max(max(MbG)));

figure(3);
imshow(uint8(MbG), 'Border', 'tight');


