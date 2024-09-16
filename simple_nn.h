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
    std::vector< std::vector<double> > layers;
    std::string actfn_name;
    std::string final_actfn_name;
    std::vector< std::vector< std::vector<double> > > weights;
    std::vector< std::vector< std::vector<double> > > gradient;
    std::vector< std::vector< std::vector<double> > > steps;

    double activation(double z);
    double actfn_derivative(double a);
    void activatefinal (std::vector<double> &finlayer);
    
    public:
    NeuralNet();
    NeuralNet(int input_size);
    NeuralNet(int input_size, int output_size);
    NeuralNet(int input_size,int output_size, int hiddenlayers);

    void getParams(std::string filename);
    void initializeParams();
    void saveParams(std::string filename);
    void initializesteps();
   
    void feedfwd(std::vector<double> &input_vec);
    std::vector<double> Output();
    double costfn( int onehotindex=0, const std::vector<double> &desired_output=std::vector<double>());
    void gradcalc( int onehothindex=0, const std::vector<double> &desired_output=std::vector<double>());
    void addtosteps(double learningrate);
    void minibatchdesc(int batch_size);

};

