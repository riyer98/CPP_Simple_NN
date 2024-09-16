#include<algorithm>
#include<iostream>
#include<vector>
#include<random>

int main(){
    std::vector<int> v1{1,4,6,2,-2,7,-3,0};
    std::vector<int> v2 = v1;

    std::random_device rd;
    std::mt19937 eng1(rd());
    std::mt19937 eng2 = eng1;

    for (int i = 0; i< 10; i++){
        std::shuffle(v1.begin(),v1.end(),eng1);
        std::shuffle(v2.begin(),v2.end(),eng2);
    }

    for (int i=0; i<8; i++) std::cout<<v1[i]<<" "<<v2[i]<<std::endl;

    return 0;
}