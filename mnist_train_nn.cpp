#include"simple_nn.h"
#include<sstream>
#include<algorithm>

using namespace std;


void getInputOutput(vector<int> &numbers, vector<vector<float> > &normedpixelvals, int &trainsize);


int main(){

    vector<int> numbers;
    vector<vector<float> > normedpixelvals;
    int trainsize=0;

    getInputOutput(numbers,normedpixelvals, trainsize);

    NeuralNet whatdigit(normedpixelvals[0].size(),10);

    whatdigit.initializeParams();

   /* clock_t start = clock();

    whatdigit.feedfwd(normedpixelvals[0]);

    vector<float> op = whatdigit.Output();

    clock_t end = clock();
    
    for(int i=0;i<10;i++) cout<<op[i]<<"\t";
    cout<<endl;
    cout<< distance(op.begin(),max_element(op.begin(),op.end()))<<endl;
    cout<<"time taken for feedfwd: "<<(float)(end-start)/1000000<<endl;

    start = clock();

    vector<float> one_hot(10,0.0);
    one_hot[numbers[0]]=1.0;

    whatdigit.gradcalc(one_hot);

    end = clock();
    cout<<"time for gradient: "<<(float)(end-start)/1000000<<endl;*/

    int epochs, batch_size, epochcount, i, batch_start, batch_end;
    int batch_remaining;
    float lossfn, accuracy;

    cout<<"Enter number of epochs: ";
    cin>>epochs;

    cout<<"Enter mini batch size: ";
    cin>>batch_size;


    random_device rd;
    mt19937 eng1(rd());
    mt19937 eng2 = eng1;

    for (epochcount=0; epochcount<epochs; epochcount++){
        batch_remaining = trainsize;
        batch_start = 0; lossfn = 0.0; accuracy =0.0;

        while(batch_start<trainsize){
            if (batch_remaining/batch_size>1)
                batch_end = batch_start+batch_size;
            else
                batch_end = batch_start+batch_remaining;
            
            vector<vector<float> > input_batch(normedpixelvals.begin()+batch_start,normedpixelvals.begin()+batch_end);
            vector<vector<float> > one_hot_outputs(input_batch.size(), vector<float>(10,0.0));

            for (i=0;i<input_batch.size();i++)
                one_hot_outputs[i][numbers[batch_start+i]]=1.0;
        
            whatdigit.minibatchdesc(input_batch, one_hot_outputs, input_batch.size());
            batch_remaining -= batch_size;
            batch_start = batch_end;
            cout<<"Success! "<<batch_end<<endl;
        }   
        
        cout<<"Epoch "<<epochcount+1<<" done.\n";
        
        for (i=0;i<trainsize;i++){
            
            whatdigit.feedfwd(normedpixelvals[i]);
            
            vector<float> output_vector = whatdigit.Output();
            
            lossfn -= log(output_vector[numbers[i]]);
            if (distance(output_vector.begin(), max_element(output_vector.begin(),output_vector.end()))==numbers[i])
                accuracy++;

        }
        lossfn/= trainsize;
        accuracy /= trainsize;
        cout<<"Accuracy = "<<accuracy<<"\t";
        cout<<"Loss = "<<lossfn<<endl;

        shuffle(normedpixelvals.begin(), normedpixelvals.end(),eng1);
        shuffle(numbers.begin(), numbers.end(),eng2);        
    }

    whatdigit.saveParams("mnist_weights.txt");

    return 0;
}



void getInputOutput(vector<int> &numbers, vector<vector<float> > &normedpixelvals, int &trainsize){
    
    ifstream trainfile("mnist_train.csv");
    string line, num; 
    cout<<"Retrieving MNIST data...\n";

    while(getline(trainfile,line)){
        stringstream s(line);

        getline(s,num,',');
        numbers.push_back(stoi(num));
        normedpixelvals.push_back(vector<float>());

        while(getline(s,num,','))
            normedpixelvals[trainsize].push_back((float)stoi(num)/255);
        trainsize++;
    }
    cout<<"Training data size = "<<trainsize<<endl;

    trainfile.close();
}
