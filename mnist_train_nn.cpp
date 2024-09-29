#include"simple_nn.h"
#include<sstream>
#include<algorithm>

using namespace std;


void getInputOutput(vector<int> &numbers, vector<vector<float> > &normedpixelvals, int &trainsize);


int main(int argc, char** argv){

    vector<int> numbers;
    vector<vector<float> > normedpixelvals;
    int trainsize=0;

    getInputOutput(numbers,normedpixelvals, trainsize);

    NeuralNet whatdigit;
   
    whatdigit.getParams(argv[1]);

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

    int epochs, batch_size, epochcount, i;
    float lossfn, accuracy, learningrate;
    vector<float> output(10);

    cout<<"Enter number of epochs: ";
    cin>>epochs;

    cout<<"Enter mini batch size: ";
    cin>>batch_size;

    cout<<"Enter learning rate: ";
    cin>>learningrate;

    random_device rd;
    mt19937 eng1(rd());
    mt19937 eng2 = eng1;

    for (epochcount=0; epochcount<epochs; epochcount++){

        shuffle(normedpixelvals.begin(), normedpixelvals.end(),eng1);
        shuffle(numbers.begin(), numbers.end(),eng2);        

        lossfn = 0.0; accuracy =0.0;

        whatdigit.initializesteps();
        for (i=0; i<trainsize; i++){

            whatdigit.feedfwd(normedpixelvals[i]);
            
            output = whatdigit.Output();
            
            lossfn += whatdigit.costfn(numbers[i]);
            if (distance(output.begin(), max_element(output.begin(),output.end()))==numbers[i]){
                accuracy++;
            }
            whatdigit.gradcalc(numbers[i]);
            whatdigit.addtosteps(learningrate);
            
            if (i%batch_size==batch_size-1){
                if(trainsize-i>batch_size){
                    whatdigit.minibatchdesc(batch_size);
                    cout<< "Mini-Batch "<<i+1<<" Successful.\n";
                    cout<<"Accuracy = "<<accuracy/(i+1)<<"\t Loss = "<<lossfn/(i+1)<<endl;
                    whatdigit.initializesteps();
                }
            }
            if(i==trainsize-1) {
                whatdigit.minibatchdesc(batch_size+trainsize%batch_size);
            }
        }
        cout<<"Epoch "<<epochcount+1<<" done.\n";
        lossfn/= trainsize;
        accuracy /= trainsize;
        cout<<"Accuracy = "<<accuracy<<"\t";
        cout<<"Loss = "<<lossfn<<endl;

    }

    whatdigit.saveParams(argv[1]);

    return 0;
}



void getInputOutput(vector<int> &numbers, vector<vector<float> > &normedpixelvals, int &trainsize){
    
    ifstream trainfile("/Users/rajgopalan/mnist_train.csv");
    string line, num; 
    cout<<"Retrieving MNIST data...\n";

    while(getline(trainfile,line)){
        stringstream s(line);

        getline(s,num,',');
        numbers.push_back(stoi(num));
        normedpixelvals.push_back(vector<float>());

        while(getline(s,num,','))
            normedpixelvals[trainsize].push_back(stof(num)/255);
        trainsize++;
    }
    cout<<"Training data size = "<<trainsize<<endl;

    trainfile.close();
}
