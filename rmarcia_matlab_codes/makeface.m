a = imread('Smile.jpg', 'JPEG');
a = uint8(rgb2gray(a));

b  = a(5:297, 5:297);
w  = conv2(double(b), ones(2));
x  = w(1:2:end, 1:2:end);
 
b = zeros(193);
b( 23:169, 23:169) = x;


imshow(uint8(b),'Border', 'tight');
% Then save as .eps file


