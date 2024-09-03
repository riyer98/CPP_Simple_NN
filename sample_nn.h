#include<vector>
#include<string>

class NeuralNet {
    private:
    int n_layers;
    std::string actfn_name;
    std::string final_actfn_name;
    int input_size;
    int output_size;

    protected:
    std::vector< std::vector< std::vector<float> > > weights;
    std::vector< std::vector<float> > layers;

    float activation(float z);
    void activatefinal (std::vector<float> &finlayer);

    public:
    void getParams(std::string filename);

    std::vector<float> getOutput(std::vector<float> &input_vec);

    void initializeParams();
    void saveParams(std::string filename);
};
