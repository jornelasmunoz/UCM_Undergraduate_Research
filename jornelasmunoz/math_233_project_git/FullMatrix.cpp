//
// Created by Matt Blomquist on 9/13/22.
//

#include "FullMatrix.h"
#include <iostream>

FullMatrix::FullMatrix() {

}

FullMatrix::FullMatrix(int N) {
    size = N;
    values.resize(size * size);

    // initialize matrix with zeros
#pragma omp parallel for
    for (int i = 0; i < size * size; i++) {
        values[i] = 0.;
    }

}

void FullMatrix::add_element(int i, int j, double v) {
    //std::cout << size << std::endl;
    if (i >= size || j >= size || i < 0 || j < 0) {
        throw std::invalid_argument("bad index value");
    } else {
        values[i + j * size] += v;
    }
}

void FullMatrix::display() {

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << values[i + j * size] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

double FullMatrix::get_value(int i, int j) {
    return values[i + j * size];
}

std::vector<double> FullMatrix::mat_Vec_Prod(std::vector<double> &x) {
    std::vector<double> prod;
    prod.resize(size);

#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        double sum = 0.;
        for (int j = 0; j < size; j++) {
            sum += get_value(i, j) * x[j];
        }
        prod[i] = sum;
    }

    return prod;
}

// Compute the matrix-matrix product
FullMatrix FullMatrix::mat_Mat_Prod(FullMatrix &B)
{
    double temp = 0.;
    int N = size;

    FullMatrix C(N);

#pragma omp parallel for
    for (int j = 0; j < N; j++){
        for (int i = 0; i < N; i++){
            temp = 0.;
            for (int k = 0; k < N; k++){
                temp += get_value(i,k) * B.get_value(k,j);
            }
            C.add_element(i,j,temp);
        }
    }

    return C;

}


// Compute the solution of Ux = b, where U is upper triangular
std::vector<double> FullMatrix::inverse_U(std::vector<double> &b)
{
    // assume the matrix is upper triangular
    // create result vector
    int N = size;

    std::vector<double> x (N);

    for (int i = N-1; i > -1; i--){
        for (int j = N-1; j > i; j--){
            x[i] += x[j] * get_value(j,i);
        }
        x[i] = (b[i] - x[i]) / get_value(i,i);
    }

    return x;
}


