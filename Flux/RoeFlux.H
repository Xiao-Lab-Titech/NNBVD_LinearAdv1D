#ifndef _ROEFLUX_H_ 
#define _ROEFLUX_H_ 
#include "../Flux.H" 
#include "../Solver.H"
#include "../Parameter.H"

class RoeFlux : public Flux 
{
	public : 
		RoeFlux(Solver* slv) : solver_(slv) {solver_->setFlux(this);}
		RoeFlux(Analyzer* analyzer) {analyzer->setFlux(this);}

		~RoeFlux(void);
		void update(std::vector<double>& flux, const std::vector<cellBoundary>& q_bdry);

		double getMCS() const {return max_characteristic_speed_;}
		std::string getName() const {std::string s = "Roe";return s;}

	private :

    private :
        double max_characteristic_speed_;
		Solver* solver_;
};


void RoeFlux::update(std::vector<double>& flux, const std::vector<cellBoundary>& q_bdry) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();

	double mws_x = 0.0;
	double a = ADV_S;
	for (int i = gs - 1; i < N_max - gs; i++) {
		double ul = q_bdry[i+1].L;
		double ur = q_bdry[i+1].R;
		flux[i+1] = 0.5*(a*ul + a*ur) - 0.5*fabs(a)*(ur - ul);
	}
	
	max_characteristic_speed_ = a;
}

#endif
