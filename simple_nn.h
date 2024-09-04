#include<vector>
#include<string>

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
    int batch_size;
    std::string graddescname;

    float activation(float z);
    float actfn_derivative(float a);
    void activatefinal (std::vector<float> &finlayer);
    float costfn(std::vector<float> output_vec);

    
    public:
    void getParams(std::string filename);
    void initializeParams();
    void saveParams(std::string filename);

    void feedfwd(std::vector<float> &input_vec);

    void gradcalc(std::vector<float> &desired_output);

};
