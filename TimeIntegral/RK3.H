#ifndef _RK3_H_
#define _RK3_H_
#include "../TimeIntegral.H"
#include "../Solver.H"
#include <vector>

class RK3 : public TimeIntegral
{
	public :
		RK3(Solver* slv);
		~RK3(void);
		void update(std::vector<double>& q, const std::vector<double>& flux);
		std::string getName() const {std::string s = "RK3";return s;}

		int getSubstep() const {return substep_;}
		int getMaxSubstep() const {return max_substep_;}
		void setSubstep(int s) {substep_ = s;}
        
    private :
		Solver* solver_;
		int substep_;
		const int max_substep_ = 3;
	
		std::vector<double> q_star_;
		std::vector<double> q_tmp_;
};



inline RK3::RK3(Solver* slv) : solver_(slv) {
	solver_->setTimeIntegral(this);

	substep_ = 1;
	int N_max = solver_->getNmax();
	q_star_ = std::vector<double>(N_max, 0.0);
	q_tmp_ = std::vector<double>(N_max, 0.0);
}



inline RK3::~RK3() {
	// std::vector members are automatically deallocated
}

void RK3::update(std::vector<double>& q, const std::vector<double>& flux) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();
	double dx = solver_->getDX();
	double dt = solver_->getDT();

	switch (substep_) {
		case 1:
			for (int i = gs; i < N_max - gs; i++) {
				q_star_[i] = q[i] + dt * (flux[i]-flux[i+1]) / dx;
				q_tmp_[i] = q[i];
				q[i] = q_star_[i];

			}
			break;
		case 2:
			for (int i = gs; i < N_max - gs; i++) {
				q[i] = 0.75*q_tmp_[i] + 0.25*q_star_[i] + 0.25 * dt * (flux[i]-flux[i+1]) / dx;
				q_star_[i] = q[i]; // q**
			}
			break;
		case 3:
			for (int i = gs; i < N_max - gs; i++) {
				q[i] = 1.0/3.0*q_tmp_[i] + 2.0/3.0*q_star_[i] + 2.0/3.0 * dt * (flux[i]-flux[i+1]) / dx;
			}
			break;
		default :
			//debagging
			break;
	}

}


#endif
