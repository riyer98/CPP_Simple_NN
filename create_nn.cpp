#include "simple_nn.h"

int main(int argc, char ** argv){
    NeuralNet ann;

    ann.initializeParams();
    ann.saveParams(argv[1]);

    return 0;
}