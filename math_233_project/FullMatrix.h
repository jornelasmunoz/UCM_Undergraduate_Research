#ifndef MATH_233_PROJECT_FULLMATRIX_H
#define MATH_233_PROJECT_FULLMATRIX_H

#include <vector>

class FullMatrix {
private:
    std::vector<double> values;
    int size;

public:
    // matrix constructors
    FullMatrix();
    FullMatrix(int N);

    // insert element v at position (i,j) in matrix
    void add_element(int i, int j, double v);

    // get values of matrix
    double get_value(int i, int j);

    // print matrix values to screen
    void display();

    // matrix vector product
    std::vector<double> mat_Vec_Prod(std::vector<double> &x);

    // matrix matrix product
    FullMatrix mat_Mat_Prod(FullMatrix &B);

    // get size of a matrix
    inline int get_Size(){ return size;};


    std::vector<double> inverse_U(std::vector<double> &b);

};


#endif //MATH_233_PROJECT_FULLMATRIX_H
