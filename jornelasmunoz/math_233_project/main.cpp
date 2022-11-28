#include <iostream>
#include <vector>
#include "FullMatrix.h"
#include "MURA.h"
#include "/usr/local/include/fftw3.h"

using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;
    // initialize MURA params
    int p = 23;
    FullMatrix A(p);
    FullMatrix G(p);

    // Create MURA objects
    MURA mura(p);
    A = mura.create_binary_aperture_arr();
    G = mura.create_decoding_arr(A);
    A.display();
    G.display();


    return 0;
}

