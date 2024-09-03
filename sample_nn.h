#include<vector>
#include<string>

using namespace std;

class NeuralNet {
    private:
    int n_layers;
    string actfn_name;
    string final_actfn_name;
    int input_size;
    int output_size;

    vector<vector<vector<float>>> weights;
    vector<vector<float>> layers;


    public:
    NeuralNet(int output_size);
    NeuralNet(int hidden_layers, int output_size);

    void getParams(string filename);

    vector<float> getOutput(vector<float> &input_vec);

    float activation(float z);
    vector<float> final_activation (vector<float> &finlayer);
    
};
