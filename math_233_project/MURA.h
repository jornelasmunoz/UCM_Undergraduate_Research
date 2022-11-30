//
// Created by Jocelyn Ornelas on 11/22/22.
//

#ifndef MATH_233_PROJECT_MURA_H
#define MATH_233_PROJECT_MURA_H

#include <vector>
#include "FullMatrix.h"

using namespace std;
class MURA {
private:
    int p;            // size of array prime number
//    FullMatrix A;
//    FullMatrix G;
public:
    int get_p(){return p;};
    // MURA mask constructors
    MURA();
    MURA(int size);

    int legendre_symbol(int a, int p);
    FullMatrix create_binary_aperture_arr();
    FullMatrix create_decoding_arr(FullMatrix &A);
};


#endif //MATH_233_PROJECT_MURA_H
