#ifndef _RK2_H_
#define _RK2_H_
#include "../TimeIntegral.H"
#include "../Solver.H"
#include <vector>

class RK2 : public TimeIntegral
{
	public :
		RK2(Solver* slv);
		~RK2(void);
		void update(std::vector<double>& q, const std::vector<double>& flux);
		std::string getName() const {std::string s = "RK2";return s;}

		int getSubstep() const {return substep_;}
		int getMaxSubstep() const {return max_substep_;}
		void setSubstep(int s) {substep_ = s;}
        
    private :
		Solver* solver_;
		int substep_;
		const int max_substep_ = 2;
	
		std::vector<double> q_tmp_;
        std::vector<double> Lh_n;
};

inline RK2::RK2(Solver* slv) : solver_(slv) {
	solver_->setTimeIntegral(this);
	substep_ = 1;
	int N_max = solver_->getNmax();
	q_tmp_ = std::vector<double>(N_max, 0.0);
	Lh_n = std::vector<double>(N_max, 0.0);
}

inline RK2::~RK2() {
	// std::vector members are automatically deallocated
}

void RK2::update(std::vector<double>& q, const std::vector<double>& flux) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();
	double dx = solver_->getDX();
	double dt = solver_->getDT();

	switch (substep_) {
		case 1:
			for (int i = gs; i < N_max - gs; i++) {
				//if (i == 0) printf("j:%d, k:%d, rho=%f\n",j,k,q[i][j][k]);
				q_tmp_[i] = q[i];
				Lh_n[i] = (flux[i]-flux[i+1])/dx;
				q[i] = q[i] + dt*Lh_n[i];
				//if (i == 0) printf("j:%d, k:%d, rho=%f\n",j,k,q[i][j][k]);
				//if (i == 0&&!std::isfinite(flux_F[i][j+1][k])) printf("j:%d, k:%d, dt=%f, flux_F_j0=%f, flux_F_j1=%f\n",j,k,dt,flux_F[i][j][k],flux_F[i][j+1][k]);
			}
			break;
		case 2:
			for (int i = gs; i < N_max - gs; i++) {
				q[i] = q_tmp_[i] + 
								+ 0.5*dt*(Lh_n[i] + (flux[i]-flux[i+1])/dx);
				//if (i == 0&&!std::isfinite(flux_F[i][j+1][k])) printf("j:%d, k:%d, dt=%f, flux_F_j0=%f, flux_F_j1=%f\n",j,k,dt,flux_F[i][j][k],flux_F[i][j+1][k]);
			}
			break;
		default :
			//debagging
			break;
	}

	//printf("rho[NX_max-gs][NY_max-gs]=%f\n",q[0][NX_max-gs][NY_max-gs]);
}

#endif
