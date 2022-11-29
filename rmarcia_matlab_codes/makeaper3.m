clear
a = imread('bcircle.jpg', 'JPEG');

bb = double(a(94:385, 175:466));
bb = 255*ones(292) - bb;
% b is 292 x 292

n  = 55;
cc = conv2(bb, ones(n));


dd = 255*cc(1:n:end, 1:n:end)/max(max(cc));

size(dd)
%return;

pn = zeros(193);

n = 3;
pn(92-n:92+n,89-n:89+n) = dd;
pn(92-n:92+n,97-n:97+n) = dd;
pn(92-n:92+n,105-n:105+n) = dd;
pn(102-n:102+n,89-n:89+n) = dd;
pn(102-n:102+n,97-n:97+n) = dd;
pn(102-n:102+n,105-n:105+n) = dd;

imshow(uint8(pn),'Border', 'tight');
% Then save as .eps file







