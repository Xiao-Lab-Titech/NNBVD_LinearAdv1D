#ifndef _SOLVER_H_
#define _SOLVER_H_
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <set>
#include <filesystem>
#include <random>


struct cellBoundary {
	double L;
	double R;
};

struct ProblemInfo {
	int i;
	std::string name;
	bool exist_exact;
};

class Reconstruction;
class Flux;
class TimeIntegral;
class BoundaryCondition;
class Exact;


class Solver 
{
	public : 
		Solver(void);
		Solver(std::string label);
		Solver(std::string label, int N_cell, int gs, double cfl);
		~Solver(void);

        void generatePreProcessedData(int N_stc);
        void generatePreProcessedDataByRandomSampling(int N_samples, int N_stc);
		void initProblem(unsigned int idx_problem, bool is_calc_exact);
		void solveUntilNextFrame(int N_frames, bool log_result, int log_period, bool gen_prep_data, int end_gen_ts, int N_stc);
		void write2File(int log_period) const;
		void initTime();
		
		std::string getLabel() const {return label_;}
		int getNcell() const {return N_cell_;}
		int getNmax() const {return N_max_;}
		int getGhostcell() const {return gs_;}
		double getCFL() const {return cfl_;}
		double getXL() const {return xl_;}
		double getXR() const {return xr_;}
		double getDX() const {return dx_;}
		double getDT() const {return dt_;}
		double getT() const {return t_;}
		double getTE() const {return te_;}
		double getTS() const {return ts_;}
		double getGamma() const {return gamma_;}
		const double* getX() const {return x_;}
		const double* getQ() const {return q_;}
		const double* getQE() const {return qe_;}
		const Reconstruction* getReconstruction() const {return recons_;}
		const Flux* getFlux() const {return fluxer_;}
		const TimeIntegral* getTimeIntegral() const {return time_integral_;}
		const BoundaryCondition* getBoundaryCondition() const {return bc_;}
		ProblemInfo getProblemInfo() const {return problem_info_;}
		//const Analyzer* getAnalyzer() const {return analyzer_;}

		void setLabel(std::string label) {label_ = label;}
		void setReconstruction(Reconstruction* recons) {recons_ = recons;}
		void setFlux(Flux* flux) {fluxer_ = flux;}
		void setTimeIntegral(TimeIntegral* ti) {time_integral_ = ti;}
		void setSelector(int idx_x, int recons, double likelifood) {selector_[idx_x][recons] = likelifood;}
		void setLogName(std::string name) {log_name_ = name;}
		void setPrepDirName(std::string name) {prep_dir_name_ = name;}

		void showInfo() const;

		void calcError() const;

	private :
		void initCell();
		void updateDT();

	private :
		std::string label_ = "";
		std::string log_name_;
		std::string prep_dir_name_;

		int N_cell_; // The number of cells
		double xl_, xr_; // left/right-side of x value
		int N_max_; // The number of cells including ghost cells
		int gs_;
		double* x_; // vector of x
		double* q_; // numerical solution. conservative variable
		double* qe_; // Exact solution. primitive variable
		double* flux_; // numerical flux
		cellBoundary* q_bdry_; // cell boundary by reconstruction
		Flux* fluxer_;
		BoundaryCondition* bc_;
		Reconstruction* recons_;
		TimeIntegral* time_integral_;
		Exact* exact_;
		double cfl_; // Courant number
		double gamma_;
		double t_; // simulating time
		double dx_;
		double dt_;
		int ts_; // time step
		double te_; // time at end of simulation
		ProblemInfo problem_info_;
		unsigned int idx_problem_;
		std::string name_of_problem_;
		bool exist_exact_;
		double** selector_; // indicating which candidate function BVD scheme selects.
		int N_cand_func_; // number of candidate functions for BVD scheme
};

#endif