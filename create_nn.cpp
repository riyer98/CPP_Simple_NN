#include "simple_nn.h"

int main(){
    NeuralNet ann;

    ann.initializeParams();
    ann.saveParams("iris_weights1.txt");

    return 0;
}