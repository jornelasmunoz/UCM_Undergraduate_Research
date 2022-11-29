a = imread('Smile.jpg', 'JPEG');
a = uint8(rgb2gray(a));

b  = a(5:297, 5:297);
w  = conv2(double(b), ones(2));
x  = w(1:2:end, 1:2:end);
 
b = zeros(193);
b( 23:169, 23:169) = x;

fb = fft2(double(b));

pn = zeros(193);
pn(97,97) = .07;

fpn  = fft2(pn);
fbpn = fb.*fpn;

c = real(ifft2(fbpn));
d = circshift(c, [97 97]);

% imwrite(uint8(d),'pinface.jpg', 'EPS');

imshow(uint8(d),'Border', 'tight');
% Then save as .eps file


