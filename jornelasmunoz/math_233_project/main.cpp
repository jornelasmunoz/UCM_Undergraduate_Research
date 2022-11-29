#include <iostream>
#include <vector>
#include "FullMatrix.h"
#include "MURA.h"
#include "/usr/local/include/fftw3.h"
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
//    int p = 23;
//    FullMatrix A(p);
//    FullMatrix G(p);

    // Create MURA objects
//    MURA mura(p);
//    A = mura.create_binary_aperture_arr();
//    G = mura.create_decoding_arr(A);
    //A.display();
    //G.display();


    int N;
    fftw_complex *in, *out; fftw_plan my_plan;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
//    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
//    my_plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//    fftw_execute(my_plan);
//    fftw_destroy_plan(my_plan);
//    fftw_free(in);
//    fftw_free(out);

//    typedef complex<double> cx;
//    cx a[] = { cx(0,0), cx(1,1), cx(3,3), cx(4,4),
//               cx(4, 4), cx(3, 3), cx(1,1), cx(0,0) };
//    cx b[8];
//    fft(a, b, 3);
////    for (int i=0; i<8; ++i)
////        cout << b[i] << "\n";
//
//    //vector<double> row;
//    vector<double> fft_res;
//    //row.resize(mura.get_p());
//
//    cx row[] = {};
//    fft_res.resize(mura.get_p());
//    int i = 0;
//    for (int j = 0; j < mura.get_p(); j++){
//        row[j] = A.get_value(i,j);
//        cout << row[j] << ",";
//    }

    return 0;
}

