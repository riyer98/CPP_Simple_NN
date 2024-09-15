#include"simple_nn.h"
#include<sstream>
//#include<algorithm>

using namespace std;


void getInputOutput(vector<vector<float> > &species, vector<vector<float> > &flowerparams, int &trainsize);


int main(int argc, char ** argv){

    vector<vector<float> > species;
    vector<vector<float> > flowerparams;
    int trainsize = 0;

    getInputOutput(species,flowerparams, trainsize);

    //NeuralNet whatspecies(flowerparams[0].size(),3);
    NeuralNet whatspecies;

    whatspecies.getParams(argv[1]);

   /*for(int i=0;i<trainsize;i++) {
        whatspecies.feedfwd(flowerparams[i]);
        vector<float> result = whatspecies.Output();
        cout<< distance(result.begin(),max_element(result.begin(),result.end()))<<"\t"<<distance(species[i].begin(),max_element(species[i].begin(),species[i].end()));
        cout<<endl;
    }*/

    //whatspecies.initializeParams();

    int epochs, batch_size, epochcount, i, batch_start, batch_end, batch_remaining;
    float lossfn, accuracy;

    cout<<"Enter number of epochs: ";
    cin>>epochs;

    cout<<"Enter mini-batch size: ";
    cin>>batch_size;

    /*random_device rd;
    mt19937 eng1(rd());
    mt19937 eng2 = eng1;*/

    for (epochcount=0; epochcount<epochs; epochcount++){

        //shuffle(flowerparams.begin(), flowerparams.end(),eng1);
        //shuffle(species.begin(), species.end(),eng2);     

        batch_remaining = trainsize;
        batch_start = 0; 
        lossfn = 0.0; accuracy = 0.0;

        /*while(batch_start<trainsize){
            if (batch_remaining/batch_size>1)
                batch_end = batch_start+batch_size;
            else
                batch_end = batch_start+batch_remaining;
            
            vector<vector<float> > input_batch(flowerparams.begin()+batch_start,flowerparams.begin()+batch_end);
            vector<vector<float> > output_batch(species.begin()+batch_start,species.begin()+batch_end);

            whatspecies.minibatchdesc(input_batch, output_batch, input_batch.size());
            batch_remaining -= batch_size;
            batch_start = batch_end;
            //cout<<"Success! "<<batch_end<<endl;
        } */

       whatspecies.minibatchdesc(flowerparams,species,trainsize);
        cout<<"Epoch "<<epochcount+1<<" done.\n";
        
        for (i=0;i<trainsize;i++){
            whatspecies.feedfwd(flowerparams[i]);
            vector<float> output = whatspecies.Output();
            //cout<<output[0]<<"\t"<<output[1]<<"\t"<<output[2]<<endl;
            lossfn += whatspecies.costfn(species[i]);
            if(distance(output.begin(),max_element(output.begin(),output.end()))==distance(species[i].begin(),max_element(species[i].begin(),species[i].end())))
                accuracy++;
        }
        lossfn/= trainsize;
        accuracy /= trainsize;
        cout<<"Accuracy = "<<accuracy<<"\t";
        cout<<"Loss = "<<lossfn<<endl;   
    }

    whatspecies.saveParams(argv[1]);

    return 0;
}



void getInputOutput(vector<vector<float> > &species, vector<vector<float> > &flowerparams, int &trainsize){
    
    ifstream trainfile("iris.csv");
    string line, num; 
    cout<<"Retrieving Iris data...\n";

    while(getline(trainfile,line)){
        stringstream s(line);

        flowerparams.push_back(vector<float>());

        while(getline(s,num,',') && num.length()<4)
            flowerparams[trainsize].push_back(stof(num));
        
        if(num=="setosa") species.push_back(vector<float>{1,0,0});
        else if (num=="versicolor") species.push_back(vector<float>{0,1,0});
        else if (num=="virginica") species.push_back(vector<float>{0,0,1});
    trainsize++;
    }
    cout<<"Training data size = "<<trainsize<<endl;

    trainfile.close();
}
