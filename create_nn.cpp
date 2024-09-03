#include "sample_nn.h"

int main(){
    NeuralNet ann;

    ann.initializeParams();
    ann.saveParams("testwts.txt");

    return 0;
}