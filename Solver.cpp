#include "Solver.H"
#include "Reconstruction.H"
#include "Flux.H"
#include "TimeIntegral.H"
//#include "BoundaryCondition.H"
#include "Parameter.H"
#include "Problem.H"
//#include "Exact.H"
//#include "Analyzer.H"
#include "RawVector.H"

FILE *fp_anim;
FILE *fp_snap;

Solver::Solver(void) {
	N_cell_ = N_CELL;
    gs_ = GHOST_CELL;
	N_max_ = N_cell_ + 2 * gs_;
	cfl_ = CFL;
	t_ = 0.0;
	dt_ = 0.0;
	ts_ = 0;
	
	initVector1d<double>(x_, N_max_, 0.0);
	initVector1d<double>(q_, N_max_, 0.0);
	initVector1d<double>(qe_, N_max_, 0.0);
	initVector1d<cellBoundary>(q_bdry_, N_max_, {0.0, 0.0});
	initVector1d<double>(flux_, N_max_, 0.0);
}

Solver::Solver(std::string label) : label_(label) {
	N_cell_ = N_CELL;
	gs_ = GHOST_CELL;
	N_max_ = N_cell_ + 2 * gs_;
	cfl_ = CFL;
	t_ = 0.0;
	dt_ = 0.0;
	ts_ = 0;
	initVector1d<double>(x_, N_max_, 0.0);
	initVector1d<double>(q_, N_max_, 0.0);
	initVector1d<double>(qe_, N_max_, 0.0);
	initVector1d<cellBoundary>(q_bdry_, N_max_, {0.0, 0.0});
	initVector1d<double>(flux_, N_max_, 0.0);
}

Solver::Solver(std::string label, int N_cell, int gs, double cfl) : label_(label), N_cell_(N_cell), gs_(gs), cfl_(cfl) {
	N_max_ = N_cell_ + 2 * gs_;
	t_ = 0.0;
	dt_ = 0.0;
	ts_ = 0;
	initVector1d<double>(x_, N_max_, 0.0);
	initVector1d<double>(q_, N_max_, 0.0);
	initVector1d<double>(qe_, N_max_, 0.0);
	initVector1d<cellBoundary>(q_bdry_, N_max_, {0.0, 0.0});
	initVector1d<double>(flux_, N_max_, 0.0);
}

Solver::~Solver(void) {
	delete bc_;
	delete exact_;
	freeVector1d<double>(x_);
	freeVector1d<double>(q_);
	freeVector1d<double>(qe_);
	freeVector1d<cellBoundary>(q_bdry_);
	freeVector1d<double>(flux_);
	freeVector2d<double>(selector_, N_max_);
}

void Solver::initTime() {
	t_ = 0.0;
	dt_ = 0.0;
	ts_ = 0;
}

void Solver::initCell(int xl, int xr) {
	dx_ = fabs(xl - xr)/N_cell_;
	for(int i = 0; i < N_max_; i++){
		x_[i] = xl + dx_*0.5 + dx_*(i-gs_);
	}
}

void Solver::setProblem(unsigned int idx_problem, bool is_calc_exact) {
	idx_problem_ = idx_problem;
	//exact_ = new Exact(this);
	//bc_ = new BoundaryCondition(this);
	N_cand_func_ = 1;
	if (recons_->getName().find("BVD")!=std::string::npos) {
		N_cand_func_ = recons_->getCandidateFunctionLabels().size();
	}
	initVector2d<double>(selector_, N_max_, N_cand_func_, 0.0);

	problem_ = new LinearAdv1DProb(this);
	problem_->setIdx(idx_problem);
	problem_->initProblem(q_, qe_);
	xl_ = problem_->getXL();
	xr_ = problem_->getXR();
	te_ = problem_->getTE();

	if (IS_USE_T_END) {
		te_ = T_END;
		problem_->setTE(T_END);
	}
}

inline void Solver::updateDT() {
	double max_characteristic_speed = fabs(fluxer_->getMCS());
    dt_ = cfl_*dx_/(max_characteristic_speed + 1.0e-15);
}

void Solver::solveUntilNextFrame(int N_frames, bool log_result, int log_period, bool gen_prep_data, int end_gen_ts, int N_stc) {
	double relative_t = 0.0;
	double plot_time_period = te_/N_frames;
	int max_substep = time_integral_->getMaxSubstep(), substep;

    while(t_ <= te_ && ((int)floor((relative_t+dt_)/plot_time_period) == (int)floor((relative_t)/plot_time_period))) {

		if (ts_ % 50 == 0 || t_ + dt_ > te_) printf("ts = %d, t = %f, dt = %f\n", ts_, t_, dt_);
		
		substep = 1;
        do {
			time_integral_->setSubstep(substep);
			bc_->update(q_);
            recons_->update(q_bdry_, q_);
			if (gen_prep_data && ts_ <= end_gen_ts && ts_ > 0 && substep == 1) {
				generatePreProcessedData(N_stc);
			}
            fluxer_->update(flux_, q_bdry_);
            updateDT();
            time_integral_->update(q_, flux_);
			//if (ts_ == 1 && substep == 1) break;

			substep++;
        } while(substep <= max_substep);
		//if (ts_ == 1) break;
		ts_ = ts_ + 1;
		t_ = t_ + dt_;
		relative_t = relative_t + dt_;
		exact_->update(qe_);

		if (log_result && log_period > 0 && (ts_ % log_period == 0 || t_ + dt_ > te_)) {
			write2File(log_period);
		} else if (log_result && log_period <= 0 && substep >= max_substep && t_ + dt_ > te_) {
			write2File(-1);
		}
	
		calcError();
    }

}



void Solver::write2File(int log_period) const {
	std::string name;
	bool is_BVD = false;
	int ss = time_integral_->getSubstep();
	int mss = time_integral_->getMaxSubstep();
	if (log_period > 0) {
		name = "./PlotData/"+log_name_+"/"+log_name_+"_ts"+std::to_string(ts_)+"_ss"+std::to_string(ss)+".dat";  
	} else if (ss >= mss) {
		name = "./PlotData/"+log_name_+"_ts"+std::to_string(ts_)+".dat";  
	}
	std::ofstream out(name);
	if (!out.is_open()) {
		std::cout  << "Fail to open:: " << name << std::endl;
		exit(EXIT_FAILURE); 
	}

	if (recons_->getName().find("BVD")!=std::string::npos) {
		is_BVD = true;
	}

	for (int i = gs_; i < N_max_ - gs_; i++) {
		//std::cout << i << " " << qe_[i] << std::endl; 
		out << ts_ << "\t" << x_[i] << "\t" << q_[i] << "\t" << qe_[i];
		out << "\t" << errors_.L1 << "\t" << errors_.L2 << "\t" << errors_.Linf;
		if (is_BVD) {
			for (int j = 0; j < N_cand_func_; j++) {
				out << "\t" << selector_[i][j];
			}
		}
		out << "\n";
	}

	out.close();
	std::cout  << name << " successfully saved. " << std::endl;
}

void Solver::generatePreProcessedData(int N_stc) { // dataset have conservative variable data.
	std::string name;
	int ss = time_integral_->getSubstep();
	std::mt19937 mt{ std::random_device{}() };
	std::string rand_str;
	static const char alphanum[] =
		"0123456789"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz";
	std::uniform_int_distribution<int> rand_char(0, sizeof(alphanum) - 2);
	rand_str.reserve(10);
	for (int i = 0; i < 10; ++i) {
		rand_str += alphanum[rand_char(mt)];
	}
	name = "./PreProcessedData/"+prep_dir_name_+"/p"+std::to_string(idx_problem_)+"_Nx"+std::to_string(N_cell_)+"_ts"+std::to_string(ts_)+"_ss"+std::to_string(ss)+"_"+rand_str+".dat";  

	std::ofstream out(name);
	if (!out.is_open()) {
		std::cout  << "Fail to open:: " << name << std::endl;
		exit(EXIT_FAILURE); 
	}

	double* q_pri, *diff_f1, *diff_f2, *diff_b2, *diff_c2, *diff_c2_2;
	initVector1d<double>(q_pri, N_stc, 0.0);
	initVector1d<double>(diff_f1, N_max_, 0.0);
	initVector1d<double>(diff_f2, N_max_, 0.0);
	initVector1d<double>(diff_b2, N_max_, 0.0);
	initVector1d<double>(diff_c2, N_max_, 0.0);
	initVector1d<double>(diff_c2_2, N_max_, 0.0);
	double M, m;
	M = *std::max_element(q_, q_ + N_max_);
	m = *std::min_element(q_, q_ + N_max_);

	for (int i = 2; i < N_max_-2; i++) {
		diff_f1[i] = q_[i+1]-q_[i];
		diff_c2[i] = (q_[i+1]-q_[i-1])*0.5;
		diff_f2[i] = -(3.0*q_[i]-4.0*q_[i+1]+q_[i+2])*0.5;
		diff_b2[i] = (q_[i-2]-4.0*q_[i-1]+3.0*q_[i])*0.5;
		diff_c2_2[i] = (q_[i-1]-2.0*q_[i]+q_[i+1]);
	}
	double M_f1, m_f1, M_f2, m_f2, M_c2, m_c2, M_b2, m_b2, M_c2_2, m_c2_2;
	M_f1 = *std::max_element(diff_f1 + 2, diff_f1 + N_max_ - 2);
	m_f1 = *std::min_element(diff_f1 + 2, diff_f1 + N_max_ - 2);
	M_f2 = *std::max_element(diff_f2 + 2, diff_f2 + N_max_ - 2);
	m_f2 = *std::min_element(diff_f2 + 2, diff_f2 + N_max_ - 2);
	M_c2 = *std::max_element(diff_c2 + 2, diff_c2 + N_max_ - 2);
	m_c2 = *std::min_element(diff_c2 + 2, diff_c2 + N_max_ - 2);
	M_b2 = *std::max_element(diff_b2 + 2, diff_b2 + N_max_ - 2);
	m_b2 = *std::min_element(diff_b2 + 2, diff_b2 + N_max_ - 2);
	M_c2_2 = *std::max_element(diff_c2_2 + 2, diff_c2_2 + N_max_ - 2);
	m_c2_2 = *std::min_element(diff_c2_2 + 2, diff_c2_2 + N_max_ - 2);


	for (int i = gs_; i < N_max_ - gs_; i++) {
		for (int j = 0; j < N_stc; j++) q_pri[j] = q_[i-(N_stc-1)/2+j];
		//int monotone_indicator = ((q_pri[(N_stc-1)/2]-q_pri[(N_stc-1)/2-1])*(q_pri[(N_stc-1)/2+1]-q_pri[(N_stc-1)/2]) < 0) ? 0 : 1;
		//int monotone_indicator = 1;
		const int dummy = 0;
		out << x_[i] << "\t";
		for (int j = 0; j < N_stc; j++) out << q_pri[j] << "\t";
		out << "\t" << M << "\t" << m << "\t" << M_f1 << "\t" << m_f1
			<< "\t" << M_f2 << "\t" << m_f2 << "\t" << M_c2 << "\t" << m_c2
			<< "\t" << M_b2 << "\t" << m_b2 << "\t" << M_c2_2 << "\t" << m_c2_2;
		for (int j = 0; j < N_cand_func_; j++) out << "\t" << selector_[i][j];
		out << "\n";
	}
	

	freeVector1d<double>(q_pri);
	freeVector1d<double>(diff_f1);
	freeVector1d<double>(diff_f2);
	freeVector1d<double>(diff_c2);
	freeVector1d<double>(diff_b2);
	freeVector1d<double>(diff_f2);
	freeVector1d<double>(diff_c2_2);
	out.close();
}


void Solver::generatePreProcessedDataByRandomSampling(int N_samples, int N_stc) {
	std::string name;
	std::filesystem::create_directory("./PreProcessedData/"+this->getLabel()+"_rand");
	name = "./PreProcessedData/"+this->getLabel()+"_rand/"+this->getLabel()+"_rand.dat";
	std::ofstream out(name);
	if (!out.is_open()) {
		std::cout  << "Fail to open:: " << name << std::endl;
		exit(EXIT_FAILURE);
	}
	if (gs_-1!=(N_stc-1)/2) {
		std::cout  << "GhostCell is set incorrectly" << std::endl;
		exit(EXIT_FAILURE);
	}
	double* q_pri;
	initVector1d<double>(q_pri, N_stc, 0.0);

	int itr = 0;
	do {
		for (int i = gs_-1; i < N_max_-gs_+1; i=i+N_stc) {
			std::mt19937 mt{ std::random_device{}() };
			std::uniform_int_distribution<int> coin(0, 1);
			std::uniform_real_distribution<double> dist(0.0, 1.0);
			int c = coin(mt);
			q_pri[0] = 0.0;
			q_pri[N_stc-1] = 1.0;
			for (int j = 1; j < N_stc-1; j++) {
				q_pri[j] = dist(mt);
				/*
				if (c == 0) {
					dist = std::uniform_real_distribution<double>(q_pri[j], 1.0);
				} else {
					dist = std::uniform_real_distribution<double>(0.0, q_pri[j]);
				}
				*/
			}
			if (c == 0) {
				std::sort(q_pri, q_pri + N_stc);
			} else {
				std::sort(q_pri, q_pri + N_stc, std::greater<double>());
			}

			for (int j = 0; j < N_stc; j++) q_[i-(N_stc-1)/2+j] = q_pri[j];
		}

		recons_->update(q_bdry_, q_);

		for (int i = gs_-1; i < N_max_-gs_+1; i=i+N_stc) {
			for (int j = 0; j < N_stc; j++) q_pri[j] = q_[i-(N_stc-1)/2+j];
			int monotone_indicator = 1;
			const int dummy = 0;
			out << dummy << "\t";
			for (int j = 0; j < N_stc; j++) out << q_pri[j] << "\t";
			out << "\t" << dummy << "\t" << dummy << "\t" << dummy << "\t" << dummy
				<< "\t" << monotone_indicator << "\t" << dummy << "\t" << dummy << "\t" << dummy << "\t" << selector_[i] << "\n";
		}
		itr += (int)(N_cell_/N_stc);
	} while(itr < N_samples);

	
	freeVector1d<double>(q_pri);
	out.close();
	std::cout  << "Successfully generated pre-processed data by random sampling." << std::endl;
}


void Solver::calcError() {
	double L1_error = 0.0, L2_error = 0.0, Linf_error = 0.0;
	int i_max=0;
	for (int i = gs_; i < N_max_ - gs_; i++) {
		double num = 0.0, exact = 0.0;
		num = q_[i];
		exact = qe_[i];
		double diff = num - exact;
		L1_error += fabs(diff);
		L2_error += diff * diff;
		if (fabs(diff) > Linf_error) {
			Linf_error = fabs(diff);
			i_max=i;
		}
	}
	L2_error = sqrt(L2_error/N_cell_);
	L1_error /= N_cell_;

	errors_.L1 = L1_error;
	errors_.L2 = L2_error;
	errors_.Linf = Linf_error;
	errors_.i_merr = i_max;
	
	//std::cout << "■ error calculated." << std::endl;
	//std::cout << "L1 error   : " << std::scientific << std::setprecision(3) << L1_error << std::endl;
	//std::cout << "L2 error   : " << std::scientific << std::setprecision(3) << L2_error << std::endl;
	//std::cout << "Linf error : " << std::scientific << std::setprecision(3) << Linf_error << ", x[i_max]: " << x_[i_max] << std::endl;
}