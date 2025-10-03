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

	Solver *slv1 = new Solver("6WBVD-d8_3000_1000_4");
	Solver *slv2 = new Solver("NNBVD");
	Solver *slv3 = new Solver("JS");
	Solver *slv4 = new Solver("Z");
	Solver *slv5 = new Solver("eta");
	Solver *slv6 = new Solver("Z+");
	Solver *slv7 = new Solver("ZA");
	Solver *slv8 = new Solver("A");



	analyzer->setSolver(slv1);
	analyzer->setSolver(slv2);
	//analyzer->setSolver(slv3);
	//analyzer->setSolver(slv4);
	//analyzer->setSolver(slv5);
	analyzer->setSolver(slv6);
	analyzer->setSolver(slv7);
	analyzer->setSolver(slv8);

	new RoeFlux(slv1);
	new RoeFlux(slv2);
	new RoeFlux(slv3);
	new RoeFlux(slv4);
	new RoeFlux(slv5);
	new RoeFlux(slv6);
	new RoeFlux(slv7);
	new RoeFlux(slv8);


	
	new RK3(slv1);
	new RK3(slv2);
	new RK3(slv3);
	new RK3(slv4);
	new RK3(slv5);
	new RK3(slv6);
	new RK3(slv7);
	new RK3(slv8);


	new nW5BVD(slv1, {"JS","Z","eta","Zplus","ZA","A"});
	new MLBasednW5BVD(slv2, onnx_path, isGPU, {"JS","Z","eta","Zplus","ZA","A"});
	new WENO5(slv3, "JS");
	new WENO5(slv4, "Z");
	new WENO5(slv5, "eta");
	new WENO5(slv6, "Zplus");
	new WENO5(slv7, "ZA");
	new WENO5(slv8, "A");



	//slv1->generatePreProcessedDataByRandomSampling(50000, 7);
	analyzer->setProblem(2);
	for (int i = 0; i < 1; i++) {
		analyzer->Solve();
	}


	//analyzer->Solve();
	//analyzer->plotSnap(-0.5,-0.1,0.9,1.1);
	analyzer->plotSnap();
	slv1->calcError();
	slv2->calcError();
	//slv3->calcError();
	//slv4->calcError();
	//slv5->calcError();
	slv6->calcError();
	slv7->calcError();
	slv8->calcError();




	return 0;  
} 
 
