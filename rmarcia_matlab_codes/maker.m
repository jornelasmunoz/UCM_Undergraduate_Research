clear

n = 256;
load fixed_h;

hmin = min(min(h));
hhat = h-hmin*ones(n,n);
g    = sqrt(hhat);
p    = real(ifft2(g));

psum = sum(abs(p(:)));
p    = p/psum;
h    = abs(fft2(p)).^2;
sh   = sum(h(:));
h    = round(h/max(h(:)));
h    = h/sum(h(:))*sh;

H    = real(fft2(h));
H(1) = 0;

a = imread('pinface0.jpg', 'JPEG');
a = double(rgb2gray(a));
b = zeros(256);
b(33:226, 33:225) = a;

Fb = fft2(b);
aFb = Fb(33:226, 33:225);
figure(5)
imshow(real(aFb), 'Border', 'tight');

figure(6)

Fbh = Fb.*H;
aFbh = Fbh(33:226, 33:225);
imshow(real(aFbh), 'Border', 'tight');

figure(7)
FFbh = real(ifft2(Fbh));
% FFbh = FFbh - min(min(FFbh));
%FFbh = FFbh + mean(a(:));
aFFbh = FFbh(33:226, 33:225);
imshow(real(FFbh), 'Border', 'tight');

figure(8)
c = conv2(FFbh, ones(2)/4);
DFFbh = c(2:2:end, 2:2:end);
d = 255*ones(256);
d(65:128+64, 65:128+64) = DFFbh

imshow(d, 'Border', 'tight');
