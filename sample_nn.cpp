#include<iostream>
#include<fstream>
#include"sample_nn.h"
#include<cmath>

NeuralNet(int output_size): n_layers(2), output_size(output_size){}

//aiming to store the params in the following format
void NeuralNet::getParams(string filename){

    ifstream paramfile=(filename);

    //top of file contains no. of layers (including input and output), activation 
    //to be used and size of input vector
    paramfile>>n_layers>>input_size>>actfn_name>>final_actfn_name;
    
    //there are n_layers-1 weight matrices. The last column of each matrix 
    //contains the biases. That is, weights[l][i][currlayersize] = b[i]
    weights.resize(n_layers-1);
    layers.resize(n_layers);
    layers[0].resize(input_size);

    int nextlayersize, currlayersize=input_size;
    
    for(int l=0;l<n_layers-1;l++){
        paramfile>>nextlayersize;
        weights[l].resize(nextlayersize);
        layers[l+1].resize(nextlayersize);
        
        for (int i=0;i<nextlayersize;i++){
            weights[l][i].resize(currlayersize+1);
            
            for (int j=0;j<currlayersize+1;j++) 
            paramfile >> weights[l][i][j];
        }
        currlayersize=nextlayersize;
    }
}


vector<float> NeuralNet::getOutput(vector<float> &input_vec){
    
    //first layer is input
    layers[0]=input_vec;
    
    //layer index, next layer index and current layer index
    int l, i, j, nextlayersize, currlayersize;
    float z;

    for (l=0; l<n_layers-1; l++){
        
        nextlayersize = weights[l].size();
        currlayersize = layers[l].size();
        
        for (i=0; i<nextlayersize; i++){
            
            z = weights[l][i][currlayersize];
            
            for (j=0;j< currlayersize;j++) {
                z += weights[l][i][j+1]* layers[l][j];
            } 
            
            if(l=n_layers-2) layers[l+1][i] = z;
            else layers[l+1][i] = activation(z); 
        }
    }
    layers[n_layers-1] = final_activation(layers[n_layers-1]);

    return layers[n_layers-1];
}


float NeuralNet::activation(float z){
    
    switch (actfn_name){
        case "relu":
        if(z<=0) return 0;
        else return z;

        case "sigmoid":
        return 1/(1+exp(-z));

        default: 
        return z;
    }
}

vector<float> final_activation(vector<float> &finlayer){

}

void NeuralNet::setWeights(){

}

