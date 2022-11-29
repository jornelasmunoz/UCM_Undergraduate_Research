#include <iostream>
#include <iomanip>
#include <vector>
#include "FullMatrix.h"
#include "MURA.h"
//#include "/usr/local/include/fftw3.h"
#include <complex>
#include <cmath>
#include <iterator>

using namespace std;

// From C++ Cookbook in O'Reilly
unsigned int bitReverse(unsigned int x, int log2n) {
    int n = 0;
    int mask = 0x1;
    for (int i=0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

const double PI = 3.1415926536;

template<class Iter_T>
void fft(Iter_T a, Iter_T b, int log2n)
{
    typedef typename iterator_traits<Iter_T>::value_type complex;
    const complex J(0, 1);
    int n = 1 << log2n;
    for (unsigned int i=0; i < n; ++i) {
        b[bitReverse(i, log2n)] = a[i];
    }
    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        complex w(1, 0);
        complex wm = exp(-J * (PI / m2));
        for (int j=0; j < m2; ++j) {
            for (int k=j; k < n; k += m) {
                complex t = w * b[k + m2];
                complex u = b[k];
                b[k] = u + t;
                b[k + m2] = u - t;
            }
            w *= wm;
        }
    }
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    // initialize MURA params
    int p = 5;
    FullMatrix A(p);
    FullMatrix G(p);

    // Create MURA objects
    MURA mura(p);
    A = mura.create_binary_aperture_arr();
    G = mura.create_decoding_arr(A);
    A.display();
    //G.display();



    typedef complex<double> cx;
    cx a[] = { cx(0,0), cx(1,1), cx(3,3), cx(4,4),
               cx(4, 4), cx(3, 3), cx(1,1), cx(0,0) };
    cx b[8];
    fft(a, b, 3);
//    for (int i=0; i<8; ++i)
//        cout << b[i] << "\n";


    // Initialize vector of complex values
    cx cx_A[p*p];
    cx fft_A[p*p];
    //cx_A.resize(p * p);
    //cx_A.assign(p * p, 0.);
    //fft_A.resize(p * p);

    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            cx_A[i * p + j ] = cx(A.get_value(i, j), 0.);
        }
    }

//    // print complex A as matrix
//    for (int i = 0; i < p; i++){
//        for (int j = 0; j < p; j++){
//            cout << cx_A[i * p + j] << setw(5);
//            //cout << cx_A[j] << setw(10);
//        }
//        cout << endl;
//    }
//    cout << endl;

    cx tmp_row[p];
    int row = 0;
    for (int j = 0; j < p; j++){
        tmp_row[j] = cx_A[row * p + j];
        cout << cx_A[row * p + j] << setw(5);

    }
    cout << endl;
    fft(tmp_row,fft_A, 3);

    for (int j = 0; j < p; ++j)
        cout << fft_A[j] << "\n";
    return 0;
}

