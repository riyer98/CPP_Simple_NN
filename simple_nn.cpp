#include"simple_nn.h"

using namespace std;

NeuralNet::NeuralNet():
input_size(0), output_size(0), n_layers(0){};

NeuralNet::NeuralNet(int input_size): 
input_size(input_size),output_size(0), n_layers(0){};

NeuralNet::NeuralNet(int input_size,int output_size): 
input_size(input_size), output_size(output_size), n_layers(0){};

NeuralNet::NeuralNet(int input_size, int output_size, int hiddenlayers):
input_size(input_size), output_size(output_size), n_layers(hiddenlayers+2){};

//aiming to store the params in the following format
void NeuralNet::getParams(string filename){

    ifstream paramfile(filename);

    //top of file contains no. of layers (including input and output), activation 
    //to be used and size of input vector
    paramfile>>n_layers>>input_size>>output_size>>actfn_name>>final_actfn_name;
    
    //there are n_layers-1 weight matrices. The last column of each matrix 
    //contains the biases. That is, weights[l][i][currlayersize] = b[i]
    weights.resize(n_layers-1);
    layers.resize(n_layers);
    gradient.resize(n_layers-1);
    steps.resize(n_layers-1);
    
    layers[0].resize(input_size);

    int nextlayersize, currlayersize=input_size;
    
    for(int l=0;l<n_layers-1;l++){
        paramfile>>nextlayersize;
        weights[l].resize(nextlayersize);
        gradient[l].resize(nextlayersize);
        steps[l].resize(nextlayersize);
        layers[l+1].resize(nextlayersize);
        
        for (int i=0;i<nextlayersize;i++){
            weights[l][i].resize(currlayersize+1);
            gradient[l][i].resize(currlayersize+1);
            steps[l][i].resize(currlayersize+1);
            
            for (int j=0;j<=currlayersize;j++) {
                paramfile >> weights[l][i][j];
            }
        }
        currlayersize=nextlayersize;
    }
    cout<<"Parameters successfully obtained.\n";
    cout<<"Input size: "<<input_size<<endl;
    cout<<"Output size: "<<output_size<<endl;
    cout<<"Hidden layers: "<<n_layers-2<<endl;
    cout<<"Activation: "<<actfn_name<<endl;
    cout<<"Final layer activation: "<<final_actfn_name<<endl;
}

void NeuralNet::initializeParams(){

    if(!input_size){
    cout<<"Enter input_size: ";
    cin>>input_size;
    }
    else cout<<"Input size is: "<<input_size<<endl;

    if(!output_size){
    cout<<"Enter output_size: ";
    cin>>output_size;
    }
    else cout<<"Output size is: "<<output_size<<endl;

    if(!n_layers){
    cout<<"Enter number of hidden layers: ";
    cin>>n_layers;
    n_layers += 2;
    }
    else cout<<"No. of hidden layers: "<<n_layers-2<<endl;

    cout<<"Enter activation function for hidden layers. Following can be used:\n";
    cout<<"1)relu    2)sigmoid    3)none(default)\n";
    cin>>actfn_name;

    cout<<"Enter activation for final output layer. Available options:\n ";
    cout<<"1)softmax    2)sigmoid   3)none(default)\n";
    cin>>final_actfn_name;
    
    int l,i,j, nextlayersize, currlayersize=input_size;

    weights.resize(n_layers-1);
    gradient.resize(n_layers-1);
    steps.resize(n_layers-1);
    layers.resize(n_layers);
    
    layers[0].resize(input_size);
    
    random_device gen;
    normal_distribution<double> dist(0.0,1.0);

    for(l=0; l<n_layers-2; l++) {
        
        cout<<"Enter no. of nodes in hidden layer "<<l+1<<": ";
        cin>>nextlayersize;
        
        weights[l].resize(nextlayersize);
        gradient[l].resize(nextlayersize);
        steps[l].resize(nextlayersize);
        layers[l+1].resize(nextlayersize);

        for (i=0;i<nextlayersize;i++){
            weights[l][i].resize(currlayersize+1);
            gradient[l][i].resize(currlayersize+1);
            steps[l][i].resize(currlayersize+1);

            for(j=0; j<currlayersize;j++){
                weights[l][i][j] = dist(gen);
            }
            
            weights[l][i][currlayersize]=0.0;
        }
        currlayersize=nextlayersize;
    }

    weights[n_layers-2].resize(output_size);
    gradient[n_layers-2].resize(output_size);
    steps[n_layers-2].resize(output_size);
    layers[n_layers-1].resize(output_size);
    
    for (i=0;i<output_size;i++){
        weights[n_layers-2][i].resize(currlayersize+1);
        gradient[n_layers-2][i].resize(currlayersize+1);
        steps[n_layers-2][i].resize(currlayersize+1);
        
        for(j=0; j<currlayersize;j++) {
            weights[n_layers-2][i][j] = dist(gen);
        }
        
        weights[n_layers-2][i][currlayersize]=0.0;
    }

    cout<<"Weights have been randomly initialized according to std normal dist.\n";
    cout<<"Biases have been initialized to 0.\n"; 
}

void NeuralNet::saveParams(string filename){
   
    ofstream paramfile(filename, ios::trunc);
    paramfile<<n_layers<<"\t\t"<<input_size<<"\t\t"<<output_size<<"\t\t"<<actfn_name<<"\t\t"<<final_actfn_name<<endl;
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
        currlayersize=nextlayersize;
    }
    paramfile.close();
    cout<<"Weights saved to "<<filename<<".\n\n";

}

//feed forward func
void NeuralNet::feedfwd(vector<double> &input_vec){
    
    //first layer is input
    layers[0]=input_vec;
    
    //layer index, next layer index and current layer index
    int l, i, j, nextlayersize, currlayersize=input_size;
    double z;
    //cout<<"feedfwd:"<<endl;
    //for(j=0;j<input_size;j++) cout<<layers[0][j]<<"  ";
    //cout<<endl;

    for (l=0; l<n_layers-1; l++){
        
        nextlayersize = weights[l].size();
        
        for (i=0; i<nextlayersize; i++){
            
            //sum initialized to bias term
            z = weights[l][i][currlayersize];
            
            for (j=0;j< currlayersize;j++) {
                //feedforward step
                z += weights[l][i][j]* layers[l][j];
            } 
            
            //final layer uses different activation (e.g. softmax)
            if(l==n_layers-2) layers[l+1][i] = z;
            else {
                layers[l+1][i] = activation(z);
                //cout<<layers[l+1][i]<<"  ";
            }
        }
        //cout<<endl;
        currlayersize=nextlayersize;
    }
    activatefinal(layers[n_layers-1]);
    //for(j=0;j<output_size;j++) cout<<layers[n_layers-1][j]<<"  ";
    //cout<<endl<<endl; 
}


vector<double> NeuralNet::Output(){
    return layers[n_layers-1];
}


double NeuralNet::activation(double z){

    if (actfn_name=="relu"){
        if(z<0) return 0.0;
        else return z;
    }

    else if (actfn_name=="sigmoid")
        return 1/(1+exp(-z));
    
    else return z;
}


void NeuralNet::activatefinal (vector<double> &finlayer){
    if (final_actfn_name=="softmax"){
        double expsum=0;
        double max = *max_element(finlayer.begin(),finlayer.end());
        
        for (int i=0;i<output_size;i++){
            finlayer[i] = exp(finlayer[i]-max);
            expsum += finlayer[i];
        }
        for (int i=0;i<output_size;i++) 
            finlayer[i] /= expsum;
    }

    else if(final_actfn_name=="sigmoid"){
        
        for (int i=0;i<output_size;i++)
            finlayer[i] = 1/(1+exp(-finlayer[i]));
    }
}


double NeuralNet::actfn_derivative(double a){
    if(actfn_name=="relu"){
        if(a==0.0) return 0.0;
        else return 1.0;
    }

    else if (actfn_name=="sigmoid")
    return a*(1-a);

    else return 1.0;
}

//the backpropagation implementation
void NeuralNet::gradcalc( int onehotindex, const vector<double> &desired_output){
    
    int l,i,j,k, prevlayersize=layers[n_layers-2].size(), currlayersize=output_size, nextlayersize;
    double gradsum;
    
    //cout<<"gradcalc"<<endl;
    //calculating gradient for weights and biases of final layer
    for (i=0;i<currlayersize;i++){
        if (desired_output.empty()){
            if (i==onehotindex) gradsum = layers[n_layers-1][i]-1.0;
            else gradsum = layers[n_layers-1][i];
        }
        else gradsum = (layers[n_layers-1][i] - desired_output[i]);

        //gradient of weights
        for(j=0;j<prevlayersize; j++){
            gradient[n_layers-2][i][j] = layers[n_layers-2][j]*gradsum;
            //cout<< gradient[n_layers-2][i][j] <<"  ";
        }

        //gradient of biases
        gradient[n_layers-2][i][prevlayersize]= gradsum;
        //cout<<gradient[n_layers-2][i][prevlayersize]<<endl;
    }
    //cout<<endl;
    nextlayersize=currlayersize;
    currlayersize=prevlayersize;
   
    //now propagating backwards
    for (l=n_layers-3; l>=0;l--){
        
        prevlayersize=layers[l].size();

        for(i=0;i<currlayersize; i++){
            gradsum=0;
            
            //summing up gradients of the forward layer to calculate gradients of
            //current layer
            for (k=0; k<nextlayersize; k++)
                gradsum += gradient[l+1][k][currlayersize]*weights[l+1][k][i];

            gradsum = gradsum * actfn_derivative(layers[l+1][i]);

            //weights gradient
            for (j=0;j<prevlayersize; j++){
                gradient[l][i][j]= layers[l][j]*gradsum;
                //cout<<gradient[l][i][j]<<"  ";
            }

            //biases gradient
            gradient[l][i][prevlayersize]=gradsum;
            //cout<<gradient[l][i][prevlayersize]<<endl;

        }
        //cout<<endl;
        nextlayersize=currlayersize;
        currlayersize=prevlayersize;
    }
}


double NeuralNet::costfn(int onehotindex, const vector<double> &desired_output){
    double result = 0; int i;

    //binary cross entropy
    //if final activation function is sigmoid.
    if (final_actfn_name=="sigmoid"){
        if (desired_output.empty()){
            return -log(layers[n_layers-1][onehotindex]);
        }
        else{
        for (i=0;i<output_size;i++)
            result += -desired_output[i]*log(layers[n_layers-1][i])- (1-desired_output[i])*log(1-layers[n_layers-1][i]);
            return result;
        }
    }
    
    //categorical cross entropy
    //use if final activation function is softmax.
    else if(final_actfn_name=="softmax"){
        if (desired_output.empty()){
            return -log(layers[n_layers-1][onehotindex]);
        }
        else{
        for (i=0;i<output_size;i++)
            result += -desired_output[i]*log(layers[n_layers-1][i]);
        return result;
        }
    }

    //default is sum of least squares.
    //use this if you do not apply activation function to final layer.
    else {
        if(desired_output.empty()){
            for (i=0;i<output_size;i++)
            if (i==onehotindex) result+= (layers[n_layers-1][i]-1.0)*(layers[n_layers-1][i]-1.0);
            else result += (layers[n_layers-1][i])*(layers[n_layers-1][i]);
        }
        else{
        for (i=0;i<output_size;i++) 
            result += (layers[n_layers-1][i]-desired_output[i])*(layers[n_layers-1][i]-desired_output[i]);
        }
        return result/2;
    }
}


void NeuralNet::initializesteps(){
    int l,i, currlayersize=input_size, nextlayersize;
    for (l=0;l<n_layers-1;l++){
        nextlayersize=steps[l].size();
        for (i=0;i<nextlayersize;i++)
            steps[l][i]=vector<double>(currlayersize+1,0.0);
        currlayersize=nextlayersize;
    }
}


void NeuralNet::addtosteps(double learningrate){

    int l, i, j, nextlayersize, currlayersize=input_size;

    for (l=0; l<n_layers-1;l++){
        nextlayersize=weights[l].size();

        for(i=0;i<nextlayersize;i++){
            for (j=0;j<=currlayersize;j++)
                steps[l][i][j]+= learningrate * gradient[l][i][j];
        }
    }
}


void NeuralNet::minibatchdesc(int batch_size){
    int l,i,j,currlayersize=input_size,nextlayersize;

    for(l=0;l<n_layers-1;l++){
        nextlayersize=weights[l].size();

        for(i=0;i<nextlayersize;i++){
            for(j=0;j<=currlayersize;j++)
                weights[l][i][j] -= steps[l][i][j]/batch_size;
        }
        currlayersize=nextlayersize;
    }
}