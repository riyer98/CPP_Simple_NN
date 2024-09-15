#include"simple_nn.h"
#include<sstream>
#include<algorithm>

using namespace std;


void getInputOutput(vector<vector<float> > &numbers, vector<vector<float> > &normedpixelvals, int &trainsize);


int main(){

    vector<vector<float> > numbers;
    vector<vector<float> > normedpixelvals;
    int trainsize=0;

    getInputOutput(numbers,normedpixelvals, trainsize);

    //NeuralNet whatdigit(normedpixelvals[0].size(),10);
    NeuralNet whatdigit;
    //whatdigit.initializeParams();
    whatdigits.getParams("mnist_weights.txt");

    /*clock_t start = clock();

    int index;

    cin>>index;

    whatdigit.feedfwd(normedpixelvals[index]);

    vector<float> op = whatdigit.Output();

    clock_t end = clock();
    
    for(int i=0;i<10;i++) cout<<op[i]<<"\t";
    cout<<endl;
    cout<<"time taken for feedfwd: "<<(float)(end-start)/1000000<<endl;

    start = clock();

    whatdigit.gradcalc(numbers[index]);

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
            if (batch_remaining/batch_size>1){
                batch_end = batch_start+batch_size;
            }
            else{
                batch_end = batch_start+batch_remaining;
            }
            
            vector<vector<float> > input_batch(normedpixelvals.begin()+batch_start,normedpixelvals.begin()+batch_end);
            vector<vector<float> > output_batch(numbers.begin()+batch_start, numbers.begin()+batch_end);
        
            whatdigit.minibatchdesc(input_batch, output_batch, input_batch.size());
            batch_remaining -= batch_size;
            batch_start = batch_end;
            //cout<<"Success! "<<batch_end<<endl;
        }   
        
        cout<<"Epoch "<<epochcount+1<<" done.\n";
        
        for (i=0;i<trainsize;i++){
            
            whatdigit.feedfwd(normedpixelvals[i]);
            
            vector<float> output = whatdigit.Output();
            
            lossfn -= whatdigit.costfn(numbers[i]);
            if (distance(output.begin(), max_element(output.begin(),output.end()))==distance(numbers[i].begin(), max_element(numbers[i].begin(),numbers[i].end()))){
                accuracy++;
            }

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



void getInputOutput(vector<vector<float> > &numbers, vector<vector<float> > &normedpixelvals, int &trainsize){
    
    ifstream trainfile("/Users/rajgopalan/mnist_train.csv");
    string line, num; 
    cout<<"Retrieving MNIST data...\n";

    while(getline(trainfile,line)){
        stringstream s(line);

        getline(s,num,',');
        numbers.push_back(vector<float>(10,0.0));
        normedpixelvals.push_back(vector<float>());

        numbers[trainsize][stoi(num)]=1.0;

        while(getline(s,num,','))
            normedpixelvals[trainsize].push_back(stof(num)/255);
        trainsize++;
    }
    cout<<"Training data size = "<<trainsize<<endl;

    trainfile.close();
}
