//
// Created by Jocelyn Ornelas on 11/22/22.
//
#include "iostream"
#include "math.h"
#include "MURA.h"
#include "FullMatrix.h"

using namespace std;
MURA::MURA(){}
MURA::MURA(int size){
    p = size;
}

// Create decoding array
FullMatrix MURA::create_binary_aperture_arr(){
//    Inputs
//    p: int. prime integer
//
//    Output
//    A: FullMatrix. Binary aperture array

    // initialize array
    FullMatrix A(get_p());

    //Aperture function p. 4350 in Gottesman and Fenimore (1989)
    for (int i = 0; i < get_p(); i++){
        for (int j = 0; j < get_p(); j++){
            int C_i;
            int C_j;
              C_i = legendre_symbol(i,get_p());
              C_j = legendre_symbol(j,get_p());
              //cout << C_i << ", " << C_j << endl;
             if (i == 0){
                 A.add_element(i,j,0);
             }
             else if (j == 0){
                 A.add_element(i,j, 1);
             }
             else if (C_i * C_j == 1){
                 A.add_element(i,j,1);
             }
             else {
                 A.add_element(i,j,0);
             }
        }
    }
    return A;
}

FullMatrix MURA::create_decoding_arr(FullMatrix &A){
//Inputs
//  A: FullMatrix. Binary aperture array
// Output
//  G: FullMatrix of same size as A. Decoding function
    // initialize array
    FullMatrix G(get_p());

    // Decoding function p. 4350 in Gottesman and Fenimore (1989)
    for (int i = 0; i < get_p(); i++){
        for (int j = 0; j < get_p(); j++){
            if (i + j == 0){
                G.add_element(i,j,1);
            }
            else if (A.get_value(i,j) == 1){
                G.add_element(i,j,1);
            }
            else if (A.get_value(i,j) == 0){
                G.add_element(i,j,-1);
            }
        }
    }
    return G;
};



int MURA::legendre_symbol(int a, int p){
    //""" Compute the Legendre symbol a|p using
    //Euler's criterion. p is a prime, a is
    //relatively prime to p (if p divides
    //a, then a|p = 0)
    int ls;
    ls = long(pow(a, int((p - 1)/2))) % p;
    return (ls == (p-1)) ? -1 : ls;
}
