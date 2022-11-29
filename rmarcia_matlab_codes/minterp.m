clear

a = imread('cameraman0.jpg');
b = double(rgb2gray(a));
c = conv2(double(b), ones(2)/4);
d = c(2:2:end, 2:2:end);

e = 255*ones(128);
e(33:96,33:96) = d;

return;

for kk = 1:128
for jj = 1:128
X(kk,jj) = 1+(jj-1)*2;
Y(jj,kk) = 1+(jj-1)*2;
    end
  end

for kk = 1:256
    for jj = 1:256
XI(kk,jj) = jj - 0.5;
YI(jj,kk) = jj - 0.5;
    end
  end

Wtheta = interp2(X,Y,d, XI, YI, 'spline');

norm(Wtheta(:) - b(:))/norm(b(:))
