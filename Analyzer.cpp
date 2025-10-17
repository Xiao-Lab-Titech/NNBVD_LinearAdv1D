#include "Analyzer.H"
#include "RawVector.H"
#include "Reconstruction.H"
#include "Flux.H"
#include "TimeIntegral.H"
#include "Problem.H"

Analyzer::Analyzer(void) {
}

Analyzer::~Analyzer(void) {

}

void Analyzer::setPlotOption(bool is_anim, int N_frames, int plot_var) {
	is_anim_ = is_anim;
	N_frames_ = N_frames;
	plot_var_ = plot_var;
}

void Analyzer::setFileNameOption(std::string disp_option, int idx_std_solver, int plot_var) {
	disp_option_ = disp_option;
	idx_std_solver_ = idx_std_solver;
	log_var_ = plot_var;	
}

std::string Analyzer::generateFileName(std::string disp_option, int idx_std_solver, int var) const {
	const Solver* std_solver = solvers_[idx_std_solver];

	// Decide what conditions to include in the file name
	// exp.) disp_option = "rftv", ext = png, var = 0 -> "p0_WENO3_HLLCFlux_RK3_var0.png" 
	// not allow ext = gif
	std::string name_of_recons, name_of_flux, name_of_timeintegral;
	const int N_cell = std_solver->getNcell();
	const double cfl = std_solver->getCFL();
	const Problem* problem = std_solver->getProblem();
	name_of_recons = std_solver->getReconstruction()->getName();
	name_of_flux = std_solver->getFlux()->getName();
	name_of_timeintegral = std_solver->getTimeIntegral()->getName();
	std::string file_name = "p"+std::to_string(problem->getIdx());

	for (int i = 0; i < disp_option.length(); i++) {
		char opt = disp_option[i]; 
		if (opt == 'P') file_name += "_"+problem->getName(); // display problem name
		if (opt == 'N') file_name += "_Nx"+std::to_string(N_cell); // display cell num.
		if (opt == 'C') file_name += "_C"+std::to_string(cfl); // display CFL num.
		if (opt == 'r') file_name += "_"+name_of_recons; // display recons.
		if (opt == 'f') file_name += "_"+name_of_flux; // display riemann solver
		if (opt == 't') file_name += "_"+name_of_timeintegral; // display Time integral
		if (opt == 'v') file_name += "_var"+std::to_string(var); // display variable
	}

	return file_name;
}


void Analyzer::setWrite2FileOption(bool log_result, int period) {
	log_result_ = log_result;
	log_period_ = period;
}

void Analyzer::setGeneratePreProcessedDataOption(bool gen_prep_data, int end_gen_ts, int N_stc) {
	gen_prep_data_ = gen_prep_data;
	end_gen_ts_ = end_gen_ts;
	N_stc_ = N_stc;
}

void Analyzer::initGnuplotOption(FILE* fp, double xl, double xr, double yb, double yt) const {
	fprintf(fp, "set xrange [%.2f:%.2f] \n", xl, xr); 
	if (yb != 0.0 || yt != 0.0) fprintf(fp, "set yrange [%.2f:%.2f] \n", yb, yt); 
	//fprintf(fp, "set xtics %.2f,0.5,%.2f  \n", xl, xr); 
	//fprintf(fp, "set ytics %.2f,0.5,%.2f  \n", ym, yM);
	fprintf(fp, "set key right outside\n");
	//fprintf(fp, "set size 1, 1\n");
	//fprintf(fp, "set term png size 750, 400\n");
	//fprintf(fp, "set palette model HSV functions gray,1,1 \n");
	fprintf(fp, "set colorsequence classic\n");
	fprintf(fp, "set grid \n");
	fprintf(fp, "set mxtics \n");
	fprintf(fp, "set mytics \n");
}

void Analyzer::plotSnap(double xl, double xr, double yb, double yt, const char* ext, int var) const {
	FILE* fp = popen("gnuplot", "w");
	bool exist_exact = solvers_[0]->getProblem()->hasExact();

	generateLabel();
    //const double xl = solvers_[0] -> getXL();
    //const double xr = solvers_[0] -> getXR();

	fprintf(fp, "set terminal %s font \"Times New Roman, 20\"\n", ext);
	fprintf(fp, "set output '%s.%s' \n", file_name_.c_str(), ext);

	if (yb != 0.0 || yt != 0.0) initGnuplotOption(fp, xl, xr, yb, yt);
	else initGnuplotOption(fp, solvers_[0]->getXL(), solvers_[0]->getXR(), yb, yt);
	fprintf(fp, "set term png size 750, 400\n");

	fprintf(fp, "plot 0 notitle lw 3.5 lc rgb 'black' ,");
	if (exist_exact) fprintf(fp, "'-' title 'Exact sol.' w line lw 2.5  lc rgb 'red',");

	int i = 1;
	for (auto slv : solvers_) {
		fprintf(fp, "'-' title '%s' w point pt %d ps 1", slv->getLabel().c_str(), i++);
		if (slv == solvers_.back()) {
			fprintf(fp, "\n");
		} else {
			fprintf(fp, ", ");
		}
	}
	if (exist_exact) {
		const int N_max = solvers_[0]->getNmax();
		const int gs = solvers_[0]->getGhostcell();
	    const double* qe = solvers_[0]->getQE();
	    const double* x = solvers_[0]->getX();

		for (int i = gs; i < N_max - gs; i++) {
			fprintf(fp, "%f %f \n", x[i], qe[i]); 
		}
		fprintf(fp, "e \n");
	}
	for (auto slv : solvers_) {
		const int N_max = slv->getNmax();
		const int gs = slv->getGhostcell();
	    const double* q = slv->getQ();
	    const double* x = slv->getX();

		for (int i = gs; i < N_max - gs; i++) {
			fprintf(fp, "%f %f \n", x[i], q[i]);
		}
		fprintf(fp, "e \n");
	}
	fflush(fp);
}



void Analyzer::plotAnim(FILE* fp, int var) const {
	generateLabel();
    const double xl = solvers_[0] -> getXL();
    const double xr = solvers_[0] -> getXR();
	bool exist_exact = solvers_[0]->getProblem()->hasExact();

	initGnuplotOption(fp, xl, xr);
	//fprintf(fp, "set terminal gif animate size 750, 400\n");
	fprintf(fp, "plot 0 notitle lw 3.5 lc rgb 'black' ,");
	if (exist_exact) fprintf(fp, "'-' title 'Exact sol.' w line lw 2.5  lc rgb 'red',");

	int i = 1;
	for (auto slv : solvers_) {
		fprintf(fp, "'-' title '%s' w point pt %d ps 1", slv->getLabel().c_str(), i++);
		if (slv == solvers_.back()) {
			fprintf(fp, "\n");
		} else {
			fprintf(fp, ", ");
		}
	}
	if (exist_exact) {
		const int N_max = solvers_[0]->getNmax();
		const int gs = solvers_[0]->getGhostcell();
	    const double* qe = solvers_[0]->getQE();
    	const double* x = solvers_[0] -> getX();

		for (int i = gs; i < N_max - gs; i++) {
			fprintf(fp, "%f %f \n", x[i], qe[i]); 
		}
		fprintf(fp, "e \n");
	}
	for (auto slv : solvers_) {
		const int N_max = slv->getNmax();
		const int gs = slv->getGhostcell();
	    const double* q = slv->getQ();
    	const double* x = slv -> getX();

		for (int i = gs; i < N_max - gs; i++) {
			fprintf(fp, "%f %f \n", x[i], q[i]);
		}
		fprintf(fp, "e \n");
	}
	fflush(fp);
}


void Analyzer::setProblem(int idx_problem) {
	idx_problem_ = idx_problem;
}

void Analyzer::initSolvers() {
	for (auto slv : solvers_) {
		slv->initTime();
		slv->setProblem(idx_problem_, true);
	}
}




void Analyzer::Solve(bool is_dry_run) {
	is_dry_run_ = is_dry_run;
        
	initSolvers();
	generateLabel();
	file_name_ = generateFileName(disp_option_ ,idx_std_solver_ , plot_var_);

	showInfo();

	if (is_dry_run_) exit(0);

	if (log_result_) {
		for(int i = 0; i < solvers_.size(); i++){
			std::string log_name = generateFileName({"PNrft"} ,i ,0);
			if (log_period_ > 0) std::filesystem::create_directory("./PlotData/"+log_name);
			solvers_[i]->setLogName(log_name);
		}
	}

	if (gen_prep_data_) {
		for(int i = 0; i < solvers_.size(); i++){
			std::string dir = solvers_[i]->getLabel(); 
			if (dir.empty()) dir = generateFileName({"rft"} ,i ,0);
			std::filesystem::create_directory("./PreProcessedData/"+dir);
			solvers_[i]->setPrepDirName(dir);
		}
	}

	FILE* fp_anim;
	if (is_anim_) {
		fp_anim = popen("gnuplot", "w");
		fprintf(fp_anim, "set terminal gif animate font \"Times New Roman, 20\" \n");
		fprintf(fp_anim, "set terminal gif animate size 750, 400\n");
		fprintf(fp_anim, "set output '%s.gif' \n", file_name_.c_str());

	}


	while (solvers_[0]->getT() <= solvers_[0]->getTE()) {
		for(auto slv : solvers_){
			slv->solveUntilNextFrame(N_frames_, log_result_, log_period_, gen_prep_data_, end_gen_ts_, N_stc_);
    	}
		//slv->getErrors();
		if (is_anim_) {
			//std::cout << "Generating animation..." << std::endl;
			plotAnim(fp_anim, plot_var_);
		}
	}
	for(auto slv : solvers_){
		std::cout << slv->getReconstruction()->getName() << " recon_time: " << slv->getReconstruction()->getReconstructionTime() << std::endl; 
	}
	showError();

	printf("\nSuccessfuly simulated!\n");
}

void Analyzer::setLabelOption(bool show_recons, bool show_flux, bool show_timeintegral) {
	show_recons_ = show_recons;
	show_flux_ = show_flux;
	show_timeintegral_ = show_timeintegral;
}

void Analyzer::generateLabel() const {
	for (auto slv : solvers_) {
		if (slv->getLabel().empty()) {
			std::string label;
			std::string recon = slv->getReconstruction()->getName();
			std::string flux = slv->getFlux()->getName();
			std::string ti = slv->getTimeIntegral()->getName();
			if (show_recons_) {
				label += recon;
			}
			if (show_flux_) {
				if (label.empty()) {
					label += flux;
				} else {
					label += "-"+flux;
				}
			}
			if (show_timeintegral_) {
				if (label.empty()) {
					label += ti;
				} else {
					label += "-"+ti;
				}
			}
			slv->setLabel(label);
		}
	}
}

void Analyzer::showInfo() const {
	std::string problem = solvers_[0]->getProblem()->getName();
	std::string run_mode = "nomal"; // nomal and dry-run
	if (is_dry_run_) run_mode = "dry";

	std::cout << "+-" << std::right << std::setw(70) << std::setfill('-') << "" << "+" << std::endl;
	std::cout << "| " << std::left << std::setw(10) << "Infomation" << std::left << std::setw(40) << std::setfill(' ') << "" << std::right << std::setw(10) << "mode:" << std::left << std::setw(10) << run_mode << "|" << std::endl;
	std::cout << "+-" << std::right << std::setw(70) << std::setfill('-') << "" << "+" << std::endl;
	std::cout << "| " << std::left << std::setw(10) << std::setfill(' ') << "Problem:" << std::left << std::setw(30) << problem << std::left << std::setw(30) << std::setfill(' ') << "" << "|" << std::endl;
	std::cout << "+-" << std::right << std::setw(70) << std::setfill('-') << "" << "+" << std::endl;
	std::cout << "| " << std::left << std::setw(15) << std::setfill(' ') << "label" << std::setw(10) << "cell_num" << std::setw(5) << "CFL" << std::setw(15) << "Recons." << std::setw(10) << "Flux" << std::setw(15) << "Time Integral" << "|" << std::endl;


	for (auto slv : solvers_) {
		std::string label = slv->getLabel();
		std::string n_cell = std::to_string(slv->getNcell());
		float cfl = slv->getCFL();

		std::string recons = slv->getReconstruction()->getName();
		std::string flux = slv->getFlux()->getName();
		std::string ti = slv->getTimeIntegral()->getName();
		
		std::cout << "| " << std::left << std::setw(15) << label << std::setw(10) << n_cell << std::setw(5) << std::fixed << std::setprecision(2) << cfl << std::setw(15) << recons << std::setw(10) << flux << std::setw(15) << ti << "|" << std::endl;
	}

	std::cout << "+-" << std::right << std::setw(70) << std::setfill('-') << "" << "+" << std::endl;
	std::cout << std::endl;
}


void Analyzer::showError() const {
	std::cout << "■ error calculated." << std::endl;
	std::cout << "L1 error   : ";
	for (auto slv : solvers_) {
		Errors err = slv->getErrors();
		std::cout << std::scientific << std::setprecision(3) << err.L1 << " ";
	}
	std::cout << std::endl;
	std::cout << "L2 error   : ";
	for (auto slv : solvers_) {
		Errors err = slv->getErrors();
		std::cout << std::scientific << std::setprecision(3) << err.L2 << " ";
	}
	std::cout << std::endl;
	std::cout << "Linf error : ";
	for (auto slv : solvers_) {
		Errors err = slv->getErrors();
		std::cout << std::scientific << std::setprecision(3) << err.Linf << " ";
	}
	std::cout << std::endl;
	
}