/**
* @file main.cpp
* @brief  the entry-point of the application usage: ./PatchTextons fname
* @author Eunbi Park
* @date 2015.11.21
*/

#include <iostream>
#include <Tools.hpp>
#include "PTexton.hpp"

using namespace std;

int main(int argc, char** argv){
	int pSize=3, K=3, knn=1;
	//string RP = "yes";
	string run = "eval";
	//patchTextons.exe data.rat K pSize RP_Type	//image_type

	if (argc < 2){
		cout << "Usage Error: Select image!" << endl;
		return 1;
	}
	else if (argc < 4){
		K = atoi(argv[2]);
	}
	else if (argc < 5){
		K = atoi(argv[2]);
		pSize = atoi(argv[3]);
	}
	else if (argc < 6){
		K = atoi(argv[2]);
		pSize = atoi(argv[3]);
		 run = argv[4];
	}

	string fname = argv[1];
	PTexton pTexton = PTexton();
	pTexton.initialize(fname, pSize, K, knn);	//file name, patch size, K,RP
		
	if (run == "eval"){
		pTexton.evaluate();
	}
	else if (run == "train"){
		pTexton.learningTexton();
		pTexton.train();
	}
	else if (run == "test"){
		pTexton.test();
		pTexton.errorAssessment();
	}
	else if (run == "assess"){
		pTexton.errorAssessment();
	}
	//else if (run == "print"){
	//	//pTexton.printTextonMap();
	//	pTexton.printResult();
	//}
	else if (run == "gray"){
		pTexton.grayscaleTexton();
	}
	return 0;
}

