#include"simple_nn.h"
#include<sstream>
#include<algorithm>

using namespace std;


void getInputOutput(vector<int> &numbers, vector<vector<float> > &normedpixelvals, int &testsize);


int main(int argc, char** argv){

    vector<int> numbers;
    vector<vector<float> > normedpixelvals;
    int testsize=0;

    getInputOutput(numbers,normedpixelvals, testsize);

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

    int i;
    float lossfn, accuracy;
    vector<float> output(10);

    while(true){
        cout<<"Enter any index of input: ";
        cin>>i;
        
        if (i<=0 || i>=testsize){
            cout<<"Error: index should be integer between 1 and "<<testsize<<endl;
            exit(1);
        }
        whatdigit.feedfwd(normedpixelvals[i-1]);
            
            output = whatdigit.Output();

            lossfn = whatdigit.costfn(numbers[i-1]);
            cout<<"Actual number = "<<numbers[i-1]<<endl;
            cout<<"Predicted = "<<distance(output.begin(),max_element(output.begin(),output.end()))<<endl;
            cout<<"Loss = "<<lossfn<<endl;
    }

    /*for (i=0; i<testsize; i++){

            whatdigit.feedfwd(normedpixelvals[i]);
            
            output = whatdigit.Output();
            
            lossfn += whatdigit.costfn(numbers[i]);
            if (distance(output.begin(), max_element(output.begin(),output.end()))==numbers[i]){
                accuracy++;
            }
            
        }
        cout<<"Testing done.\n";
        lossfn/= testsize;
        accuracy /= testsize;
        cout<<"Accuracy = "<<accuracy<<"\t";
        cout<<"Loss = "<<lossfn<<endl;*/

    return 0;
}



void getInputOutput(vector<int> &numbers, vector<vector<float> > &normedpixelvals, int &testsize){
    
    ifstream testfile("/Users/rajgopalan/mnist_test.csv");
    string line, num; 
    cout<<"Retrieving MNIST data...\n";

    while(getline(testfile,line)){
        stringstream s(line);

        getline(s,num,',');
        numbers.push_back(stoi(num));
        normedpixelvals.push_back(vector<float>());

        while(getline(s,num,','))
            normedpixelvals[testsize].push_back(stof(num)/255);
        testsize++;
    }
    cout<<"testing data size = "<<testsize<<endl;

    testfile.close();
}