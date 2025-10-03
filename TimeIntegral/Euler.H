#ifndef _Euler_H_
#define _Euler_H_
#include "../TimeIntegral.H"
#include "../Solver.H"
#include "../RawVector.H"

class Euler : public TimeIntegral
{
	public :
		Euler(Solver* slv);
		~Euler(void);
		void update(double* q, const double* flux);
		std::string getName() const {std::string s = "Euler";return s;}

		int getSubstep() const {return substep_;}
		int getMaxSubstep() const {return max_substep_;}
		void setSubstep(int s) {substep_ = s;}
        
    private :
		Solver* solver_;
		int substep_;
		const int max_substep_ = 1;
	
};

inline Euler::Euler(Solver* slv) : solver_(slv) {
	solver_->setTimeIntegral(this);

	substep_ = 1;
}

inline Euler::~Euler() {}

void Euler::update(double* q, const double* flux) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();
	double dx = solver_->getDX();
	double dt = solver_->getDT();

	switch (substep_) {
		case 1:
			for (int i = gs; i < N_max - gs; i++) {
				q[i] = q[i] + dt * (flux[i]-flux[i+1]) / dx;
			}
			break;
		default :
			//debagging
			break;
	}

}


#endif
