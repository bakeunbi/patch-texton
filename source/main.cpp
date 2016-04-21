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
   
	int pSize, K;
	string RP = "no";
	//patchTextons.exe data.rat K pSize RP_Type	//image_type

	if (argc < 2){
		cout << "Usage Error: Select image!" << endl;
		return 1;
	}
	else if (argc < 3){
		K = 10;
		pSize = 5;
	}
	else if (argc < 4){
		K = atoi(argv[2]);
	}
	else if (argc < 5){
		pSize = atoi(argv[3]);
	}
	else if (argc < 6){
		 RP= argv[4];
	}

	string fname = argv[1];
	PTexton pTexton(fname,pSize,K,RP);	//file name, patch size, K,RP
	
	//pTexton.learningTexton();
	//pTexton.train();
	//pTexton.test();
	pTexton.evaluate();

	return 0;
}

