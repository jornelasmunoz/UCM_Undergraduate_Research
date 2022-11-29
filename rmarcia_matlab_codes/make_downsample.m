clear 
a = imread('Smile.jpg', 'JPEG');
a = uint8(rgb2gray(a));

b  = a(5:297, 5:297);
w  = conv2(double(b), ones(2));
x  = w(1:2:end, 1:2:end);
 
b = zeros(193);
b( 23:169, 23:169) = x;

c = conv2(b, ones(2))/4;
cc = c(2:2:end, 2:2:end);
d = 256*ones(193);
d(49:49+96, 49:49+96) = cc;




imshow(uint8(d),'Border', 'tight');
% Then save as .eps file


