#include <iostream>
#include <iomanip>
#include <vector>
#include "FullMatrix.h"
#include "MURA.h"
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
    //A.display();
    //G.display();



    typedef complex<double> cx;
    // Initialize list of complex values
    cx cx_A[p*p];
    cx fft_A_row[p*p];
    cx fft_A_col[p*p];
    cx tmp_row[p];
    cx tmp_res[p];
    cx tmp_col[p];
    cx tmp_res_col[p];

    // Fill complex A with values from A
    // essentially just putting the numbers in complex form
    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            cx_A[i * p + j ] = cx(A.get_value(i, j), 0.);
        }
    }

    for (int row = 0; row < p; row++) {
        //int row = 1;
        // get one row from complex A
        for (int j = 0; j < p; j++) {
            tmp_row[j] = cx_A[row * p + j];
            //cout << cx_A[row * p + j] << setw(5);
        }
        cout << endl;
        // do FFT of row
        fft(tmp_row, tmp_res, 3);

        // store fft results in larger matrix
        for (int j = 0; j < p; j++) {
            fft_A_row[row * p + j] = tmp_res[j];
            //cout << cx_A[row * p + j] << setw(5);
        }
    }
//    for (int j = 0; j < p*p; ++j)
//        cout << fft_A[j] << "\n";

    // COLUMNS
    int col = 2;
    // define temporary column array
    for (int j = 0; j < p; j++) {
        tmp_col[j] = fft_A_row[col + j * p ];
        //cout << col + j * p << endl;
        //cout << tmp_col[j] << " ";
    }
    cout << endl;
    // do FFT of row
    fft(tmp_col, tmp_res_col, 3);

    // store fft results in larger matrix
    for (int j = 0; j < p; j++) {
        fft_A_col[col + j * p] = tmp_res_col[j];
        //cout << col + j * p << endl;
        //cout << cx_A[row * p + j] << setw(5);
    }


     //print  matrix
    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            cout << fft_A_row[i * p + j] << "    ";
            //cout << cx_A[j] << setw(10);
        }
        cout << endl;
    }
    cout << endl;

    //print  matrix
    for (int i = 0; i < p; i++){
        for (int j = 0; j < p; j++){
            cout << fft_A_col[i * p + j] << "    ";
            //cout << cx_A[j] << setw(10);
        }
        cout << endl;
    }
    cout << endl;
    return 0;
}

