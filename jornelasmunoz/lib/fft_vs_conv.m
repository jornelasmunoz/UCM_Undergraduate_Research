clc; clear all;
p=5;
A = [0, 0, 0, 0, 0; 
    1, 1, 0, 0, 1; 
    1, 0, 1, 1, 0; 
    1, 0, 1, 1, 0; 
    1, 1, 0, 0, 1];%randn(3);
B = [-1, -1, -1, -1, -1; 
    1, 1, -1, -1, 1; 
    1, -1, 1, 1, -1; 
    1, -1, 1, 1, -1; 
    1, 1, -1, -1, 1];%randn(3);
 
fprintf('\nConvolution using conv2. C = conv2(A,B): \n')
C = conv2(A,B)  
 
fprintf('Convolution using FFT with "zero" padding of width/height 5.\n')
F = real(ifft2(fft2(A,9,9).*fft2(B,9,9))) 
 
fprintf('|| C - F ||_F = %12.8e\n', norm(C-F,'fro'))  
 
fprintf('\nConvolution using conv2 but using a repeating B.\n')
BB = [B B; B B];
 
C9 = conv2(BB,A)
%  
C3 = C9(p+1:2*p, p+1:2*p)%(4:6, 4:6)
%  
fprintf('Convolution using FFT without padding.\n')
F3 = real(ifft2(fft2(A).*fft2(B))) 
%circshift(F3, [(p-1)/2  (p-1)/2])
  
fprintf('|| C3 - F3 ||_F = %12.8e\n', norm(C3-F3,'fro'))  