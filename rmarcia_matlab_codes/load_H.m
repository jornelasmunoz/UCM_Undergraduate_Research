load  fixed_h;


optn = 2;


% Option 1
if (optn == 1)
p = ones(n)/(n^2);
h = abs(fft2(p)).^2;
H = real(fft2(h));
end

% Option 2
if (optn == 2)
h = h/sum(abs(h(:)));
H = real(fft2(h))*256;
end


% Option 3
if (optn == 3)
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
end


% Option 4
if (optn == 4)
hmin = min(min(h));
hhat = h-hmin*ones(n,n);
g    = sqrt(hhat);
p    = real(ifft2(g));
  
p    = sign(p);
psum = sum(abs(p(:)));
p    = p/psum;
p    = 4*p;
hb   = abs(fft2(p)).^2;
H    = real(fft2(hb)); 
H(1) = 0;
end


