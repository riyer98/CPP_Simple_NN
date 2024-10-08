#include"simple_nn.h"
#include<sstream>
#include<algorithm>

using namespace std;


void getInputOutput(vector<int> &species, vector<vector<float> > &flowerparams, int &datasize);


int main(int argc, char ** argv){

    vector<int> species;
    vector<vector<float> > flowerparams;
    int datasize = 0;

    getInputOutput(species,flowerparams, datasize);

    int trainsize=(int)(0.8*datasize);
    //int trainsize=datasize;

    cout<<"Total data size = "<<datasize<<endl;
    cout<<"Data size used for training = "<<trainsize<<endl;

    NeuralNet whatspecies;

    whatspecies.getParams(argv[1]);

    int epochs, batch_size, epochcount, i;
    float lossfn, accuracy, learningrate=0.1;
    vector<float> output;

    cout<<"Enter number of epochs: ";
    cin>>epochs;

    cout<<"Enter mini-batch size: ";
    cin>>batch_size;

    random_device rd;
    mt19937 eng1(rd());
    mt19937 eng2 = eng1;

    for(i=0;i<100;i++){
        shuffle(flowerparams.begin(), flowerparams.end(),eng1);
        shuffle(species.begin(), species.end(),eng2);
    }

    for (epochcount=0; epochcount<epochs; epochcount++){
    
        shuffle(flowerparams.begin(), flowerparams.begin()+trainsize,eng1);
        shuffle(species.begin(), species.begin()+trainsize,eng2);     

        lossfn = 0.0; accuracy = 0.0;

        whatspecies.initializesteps();
        
        for (i=0;i<trainsize;i++){

            //cout<<flowerparams[i][0]<<"\t"<<flowerparams[i][1]<<"\t"<<flowerparams[i][2]<<"\t"<<flowerparams[i][3]<<"\n";
            //cout<<species[i]<<endl;

            whatspecies.feedfwd(flowerparams[i]);
            //cout<<"feedfwd successful "<<i<<endl;
            
            output = whatspecies.Output();
            //cout<<output[0]<<"\t"<<output[1]<<"\t"<<output[2]<<"\n";
            
            lossfn += whatspecies.costfn(species[i]);
            if (distance(output.begin(), max_element(output.begin(),output.end()))==species[i]){
                accuracy++;
            }
            whatspecies.gradcalc(species[i]);
            //cout<<"gradcalc successful "<<i<<endl;
            
            whatspecies.addtosteps(learningrate);
            //cout<<"addtosteps successful "<<i<<endl;
            
            if (i%batch_size==batch_size-1){
                if(trainsize-i>batch_size){
                    whatspecies.minibatchdesc(batch_size);
                    //cout<<"MINI BATCH DESCENT SUCCESS"<<i<<endl;
                    whatspecies.initializesteps();
                    //cout<<"INITSTEPS SUCCESS "<<i<<endl;
                }
            }
            if(i==trainsize-1) {
                whatspecies.minibatchdesc(batch_size+trainsize%batch_size);
            }
        }
        cout<<"Epoch "<<epochcount+1<<" done.\n";
        lossfn/= trainsize;
        accuracy /= trainsize;
        cout<<"Accuracy = "<<accuracy<<"\t";
        cout<<"Loss = "<<lossfn<<endl;  
    }

   cout<<"now testing\n";
    lossfn=0; accuracy=0;
    for(i=trainsize;i<datasize;i++){
        whatspecies.feedfwd(flowerparams[i]);
        output=whatspecies.Output();
        lossfn += whatspecies.costfn(species[i]);
            if (distance(output.begin(), max_element(output.begin(),output.end()))==species[i]){
                accuracy++;
            }
    }
    lossfn/= (datasize-trainsize);
    accuracy /= (datasize-trainsize);
    cout<<"Accuracy = "<<accuracy<<"\t";
    cout<<"Loss = "<<lossfn<<endl;

    whatspecies.saveParams(argv[1]);

    return 0;
}



void getInputOutput(vector<int> &species, vector<vector<float> > &flowerparams, int &datasize){
    
    ifstream trainfile("iris.csv");
    string line, num; 
    cout<<"Retrieving Iris data...\n";

    while(getline(trainfile,line)){
        stringstream s(line);

        flowerparams.push_back(vector<float>());

        while(getline(s,num,',') && num.length()<4)
            flowerparams[datasize].push_back(stof(num));
        
        if(num=="setosa") species.push_back(0);
        else if (num=="versicolor") species.push_back(1);
        else if (num=="virginica") species.push_back(2);
    datasize++;
    }

    trainfile.close();
}
