clear
a = imread('bcircle.jpg', 'JPEG');

bb = double(a(94:385, 175:466));
bb = 255*ones(292) - bb;
% b is 292 x 292

cc = 2.2*conv2(bb, ones(30,30))/90000000; 


dd =  cc(1:30:end, 1:30:end);



a = imread('Smile.jpg', 'JPEG');
a = uint8(rgb2gray(a));

b  = a(5:297, 5:297);
w  = conv2(double(b), ones(2));
x  = w(1:2:end, 1:2:end);

b = zeros(193);
b( 23:169, 23:169) = x;

fb = fft2(double(b));

pn = zeros(193);


pn(97-5:97+5,97-5:97+5) = dd;

fpn  = fft2(pn);
fbpn = fb.*fpn;

c = real(ifft2(fbpn));
d = circshift(c, [97 97]);

imshow(uint8(d),'Border', 'tight');
% Then save as .eps file







