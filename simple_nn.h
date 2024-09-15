#include<vector>
#include<string>
#include<iostream>
#include<fstream>
#include<cmath>
#include<random>

class NeuralNet {

    private:
    int n_layers;
    int input_size;
    int output_size;
    std::vector< std::vector< std::vector<float> > > weights;
    std::vector< std::vector<float> > layers;
    std::string actfn_name;
    std::string final_actfn_name;
    
    std::vector< std::vector< std::vector<float> > > gradient;
    std::vector< std::vector< std::vector<float> > > steps;

    float activation(float z);
    float actfn_derivative(float a);
    void activatefinal (std::vector<float> &finlayer);
    
    public:
    NeuralNet();
    NeuralNet(int input_size);
    NeuralNet(int input_size, int output_size);
    NeuralNet(int input_size,int output_size, int hiddenlayers);

    void getParams(std::string filename);
    void initializeParams();
    void saveParams(std::string filename);
    void initializesteps();
   
    void feedfwd(std::vector<float> &input_vec);
    std::vector<float> Output();
    float costfn(std::vector<float> &desired_output);
    void gradcalc(std::vector<float> &desired_output);
    void addtosteps(std::vector<float> &input_vec, std::vector<float> &desired_output);
    void minibatchdesc(std::vector<std::vector<float> > &input_batch, std::vector<std::vector<float> > &output_batch, int batch_size);

};

