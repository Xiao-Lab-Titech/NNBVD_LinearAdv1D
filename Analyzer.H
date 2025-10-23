#ifndef _ANALYZER_H_
#define _ANALYZER_H_
#include "Solver.H"
#include <vector>


class Analyzer 
{
	public : 
		Analyzer(void);
		~Analyzer(void);

		void setPlotOption(bool is_anim = false,
						   int N_frames = 1,
						   int plot_var = 0);
		void setFileNameOption(std::string disp_option = "N",
							   int idx_standard_solver = 0,
							   int plot_var = 0);
		void setLabelOption(bool show_recons = true,
						    bool show_flux = false,
							bool show_timeintegral = false);
		void setWrite2FileOption(bool log_result = false,
								 int period = 25);
		void setGeneratePreProcessedDataOption(bool gen_prep_data = false,
											   int end_gen_ts = 50,
											   int N_stc = 7);

		//const std::vector<Solver*> getSolvers() const {return solvers_;}

		void setReconstruction(Reconstruction* recons) {for (auto slv : solvers_) slv->setReconstruction(recons);}
		void setFlux(Flux* flux) {for (auto slv : solvers_) slv->setFlux(flux);}
		void setTimeIntegral(TimeIntegral* ti) {for (auto slv : solvers_) slv->setTimeIntegral(ti);}
		void setSolver(Solver* slv) {solvers_.push_back(slv);}


		void plotSnap(double xl=0.0, double xr=0.0, double yb=0.0, double yt=0.0, const char* ext = "png", int var = 0) const;
		void setProblem(int idx_problem);
		void Solve(bool is_dry_run = false);

		void showInfo() const;
		void showError() const;


	private :
		void generateLabel() const;
		std::string generateFileName(std::string disp_option, int idx_std_solver, int plot_var) const;
		void initSolvers();
		void plotAnim(FILE* fp, int var = 0) const;
		void initGnuplotOption(FILE* fp, double xl, double xr, double yb=0.0, double yt=0.0) const;


	private :
		std::vector<Solver*> solvers_;


		std::string file_name_ = "graph";
		int idx_problem_;

		// Plot option
		bool is_anim_ = false;
		int N_frames_ = 1;
		int plot_var_ = 0;

		// Filename option
		std::string disp_option_;
		int idx_std_solver_;
		int log_var_;

		// label option
		bool show_recons_ = true;
		bool show_flux_ = false;
		bool show_timeintegral_ = false;

		// write2File option
		bool log_result_ = false;
		int log_period_ = 25;

		// generatePreProcessedData option
		bool gen_prep_data_ = false;
		int end_gen_ts_ = 50;
		int N_stc_ = 7;

		bool is_dry_run_ = false;

};

#endif