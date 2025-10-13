#include "Solver.H"
#include "Analyzer.H"
//#include "./Reconstruction/W3TBVD.H"
//#include "./Reconstruction/MTBVD.H"
#include "./Reconstruction/nW5BVD.H"
//#include "./Reconstruction/WENO3.H"
#include "./Reconstruction/WENO5.H"
//#include "./Reconstruction/THINC.H"
#include "./Reconstruction/Upwind.H"
//#include "./Reconstruction/Polynominal.H"
//#include "./Reconstruction/MUSCL.H"
//#include "./Reconstruction/MLBasedW3TBVD.H"
#include "./Reconstruction/MLBasednW5BVD.H"
//#include "./Reconstruction/MLBasedMTBVD.H"
#include "./Flux/RoeFlux.H"
#include "./TimeIntegral/RK3.H"
#include "./TimeIntegral/RK2.H"
#include "./TimeIntegral/Euler.H"
#include <chrono>


int main (int argc, char *argv[]) 
{
	const char* onnx_path = "./ONNX/d15_3000_1000_4.onnx";
	bool isGPU = true;

	Analyzer* analyzer = new Analyzer();
	analyzer->setPlotOption(false, 30, 0);
	analyzer->setFileNameOption("N");
	analyzer->setLabelOption(true, false, false);
	analyzer->setWrite2FileOption(false, -1);
	analyzer->setGeneratePreProcessedDataOption(false, 1000, 7);

	Solver *slv1 = new Solver("org. BVD");
	Solver *slv2 = new Solver("Strict1 BVD");
	Solver *slv3 = new Solver("Strict2 BVD");
	Solver *slv4 = new Solver("Strict3 BVD");
	Solver *slv5 = new Solver("WENO-ZA");


	analyzer->setSolver(slv1);
	analyzer->setSolver(slv2);
	analyzer->setSolver(slv3);
	analyzer->setSolver(slv4);
	//analyzer->setSolver(slv5);

	new RoeFlux(slv1);
	new RoeFlux(slv2);
	new RoeFlux(slv3);
	new RoeFlux(slv4);
	new RoeFlux(slv5);

	
	new RK3(slv1);
	new RK3(slv2);
	new RK3(slv3);
	new RK3(slv4);
	new RK3(slv5);


	new nW5BVD(slv1, {"JS","Z","eta","Zplus","ZA","A"}, 0, false);
	//new MLBasednW5BVD(slv2, onnx_path, isGPU, {"JS","Z","eta","Zplus","ZA","A"});
	new nW5BVD(slv2, {"JS","Z","eta","Zplus","ZA","A"}, 2, false);
	new nW5BVD(slv3, {"JS","Z","eta","Zplus","ZA","A"}, 1, false);
	new nW5BVD(slv4, {"JS","Z","eta","Zplus","ZA","A"}, 3, false);
	new WENO5(slv5, "ZA");



	//slv1->generatePreProcessedDataByRandomSampling(50000, 7);
	analyzer->setProblem(8);
	for (int i = 0; i < 1; i++) {
		analyzer->Solve();
	}


	//analyzer->Solve();
	analyzer->plotSnap(-0.1,0.1,-0.01,0.01);
	//analyzer->plotSnap();
	slv1->calcError();
	slv2->calcError();
	slv3->calcError();
	slv4->calcError();
	//slv5->calcError();
	//slv6->calcError();
	//slv7->calcError();
	//slv8->calcError();




	return 0;  
} 
 
