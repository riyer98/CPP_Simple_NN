#include<iostream>
#include<fstream>
#include"sample_nn.h"
#include<cmath>
#include<random>

using namespace std;

//aiming to store the params in the following format
void NeuralNet::getParams(string filename){

    ifstream paramfile(filename);

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
                z += weights[l][i][j]* layers[l][j];
            } 
            
            if(l==n_layers-2) layers[l+1][i] = z;
            else layers[l+1][i] = activation(z); 
        }
    }
    activatefinal(layers[n_layers-1]);

    return layers[n_layers-1];
}


float NeuralNet::activation(float z){

    if (actfn_name=="relu"){
        if(z<=0) return 0;
        else return z;
    }

    else if (actfn_name=="sigmoid")
        return 1/(1+exp(-z));
    
    else return z;
}

void NeuralNet::activatefinal (vector<float> &finlayer){
    if (final_actfn_name=="softmax"){
        float expsum=0;
        
        for (int i=0;i<finlayer.size();i++){
            finlayer[i] = exp(finlayer[i]);
            expsum += finlayer[i];
        }
        for (int i=0;i<finlayer.size();i++) 
            finlayer[i] /= expsum;
    }

    else if(final_actfn_name=="sigmoid"){
        for (int i=0;i<finlayer.size();i++)
            finlayer[i] = 1/(1+exp(-finlayer[i]));
    }
}

void NeuralNet::initializeParams(){
    
    cout<<"Enter input_size: ";
    cin>>input_size;

    cout<<"Enter output_size: ";
    cin>>output_size;

    cout<<"Enter number of hidden layers: ";
    cin>>n_layers;
    n_layers += 2;

    cout<<"Enter activation function for hidden layers. Following can be used:\n";
    cout<<"1)relu    2)sigmoid    3)none(default)\n";
    cin>>actfn_name;

    cout<<"Enter activation for final output layer. Available options:\n ";
    cout<<"1)softmax    2)sigmoid\n";
    cin>>final_actfn_name;
    
    int l,i,j, nnodes, currsize=input_size;

    weights.resize(n_layers-1);
    
    random_device gen;
    normal_distribution<float> dist(0.0,1.0);

    for(l=0; l<n_layers-2; l++) {
        cout<<"Enter no. of nodes in hidden layer "<<l+1<<": ";
        cin>>nnodes;
        weights[l].resize(nnodes);

        cout<<weights[l].size()<<endl;

        for (i=0;i<nnodes;i++){
            weights[l][i].resize(currsize+1);
            
            for(j=0; j<currsize;j++)
                weights[l][i][j] = dist(gen);
            
            weights[l][i][currsize]=0;
        }
        currsize=nnodes;
    }

    weights[n_layers-2].resize(output_size);
    
    for (i=0;i<output_size;i++){
        weights[n_layers-2][i].resize(currsize+1);
        
        for(j=0; j<currsize;j++) 
            weights[n_layers-2][i][j] = dist(gen);
        
        weights[n_layers-2][i][currsize]=0;
        }

    cout<<"Weights have been randomly initialized according to std normal dist.\n";
    cout<<"Biases have been initialized to 0.\n"; 
}

void NeuralNet::saveParams(string filename){
   
    ofstream paramfile(filename, ios::trunc);
    paramfile<<n_layers<<"\t\t"<<input_size<<"\t\t"<<actfn_name<<"\t\t"<<final_actfn_name<<endl;
    paramfile<<endl;

    int l,i,j, nextlayersize, currlayersize=input_size;
    for(l=0;l<n_layers-1;l++){
        nextlayersize=weights[l].size();

        paramfile<<nextlayersize<<endl;
        
        for(i=0;i<nextlayersize; i++){
            
            for(j=0;j<currlayersize+1;j++)
            paramfile<<weights[l][i][j]<<"\t\t"; 
            
            paramfile<<endl;
        }
        paramfile<<endl;
    }
    paramfile.close();
    cout<<"Save succesful.\n";

}
