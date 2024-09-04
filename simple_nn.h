#include<vector>
#include<string>

class NeuralNet {
    
    protected:
    int n_layers;
    int input_size;
    int output_size;
    std::vector< std::vector< std::vector<float> > > weights;
    std::vector< std::vector<float> > layers;
    std::string actfn_name;
    std::string final_actfn_name;

    float activation(float z);
    void activatefinal (std::vector<float> &finlayer);

    
    public:
    void getParams(std::string filename);
    void initializeParams();
    void saveParams(std::string filename);

    void feedfwd(std::vector<float> &input_vec);

};


class trainer: public NeuralNet {
    private:
    std::vector< std::vector< std::vector<float> > > gradient;
    int batch_size;
    std::string graddescname;
    
    public:
    float costfn(std::vector<float> output_vec);
    float actfn_derivative(float a);

    void backprop(std::vector<float> &desired_output);
};