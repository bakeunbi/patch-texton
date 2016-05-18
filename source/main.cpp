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

	string fname = argv[1];
	if (fname == "help"){
		cout << "<Usage> \n"
			<<"Train: patchTextons.exe filename(.rat) train k fsize \n"
			<< "Test: patchTextons.exe filename(.rat) test k pSize knn\n\n"
			<< "<Parameter description>\n"
			<< "k: the parameter deciding the number of center in each class of a fold\n"
			<< "\t(Finally, the number of textons is 4fold x 5 classes x k = 20k)\n"
			<< "fsize: the parameter deciding the feature dimension\n"
			<< "\t(Finally, feature dimension is fsize x fsize\n"
			<< "pSize: the parameter deciding the patch size for learning histogram\n"
			<< "\t(Finally, the patch size is pSize x pSize.\n"
			<< "knn: the parameter for k-nn\n\n"
			<< "<default parameter setting>\n"
			<< "k = 3, pSize = 3, knn = 1\n";
	}

	if (argc > 2){
		PTexton pTexton = PTexton();
		int pSize = 3, K = 3, knn = 1;

		string run = argv[2];
		
		if (run == "train"){
			if (argc > 4){
				K = atoi(argv[3]);
				pSize = atoi(argv[4]);
				pTexton.initialize(fname, pSize, K, knn);	//file name, (fsize or pSize), K,knn
				pTexton.learningTexton();
				pTexton.train();
			}
			else{
				cout << "Usage Error: \"help\"" << endl;
				return 1;
			}
		}
		//else if (run == "gray"){
		//	if (argc > 4){
		//		K = atoi(argv[3]);
		//		pSize = atoi(argv[4]);
		//		pTexton.initialize(fname, pSize, K, knn);	//file name, (fsize or pSize), K,knn
		//		pTexton.grayscaleTexton();
		//	}
		//	else{
		//		cout << "Usage Error: \"help\"" << endl;
		//		return 1;
		//	}
		//	
		//}
		else if (run == "test"){
			if (argc > 5){
				K = atoi(argv[3]);
				pSize = atoi(argv[4]);
				knn = atoi(argv[5]);

				pTexton.initialize(fname, pSize, K, knn);	//file name, (fsize or pSize), K,knn
				pTexton.test();
				pTexton.errorAssessment();
			}
			else{
				cout << "Usage Error: \"help\"" << endl;
				return 1;
			}
		}
	}

	if (argc < 2){
		cout << "Usage Error: Select image!" << endl;
		return 1;
	}

	return 0;
}

