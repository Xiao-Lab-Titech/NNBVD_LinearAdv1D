#ifndef _FLUX_H_
#define _FLUX_H_ 
//#include "Solver.H"
#include <vector>

class Solver;
//class Analyzer;

class Flux 
{
	public : 
		Flux(void){}
		Flux(Solver* slv) : solver_(slv) {solver_->setFlux(this);}
		//Flux(Analyzer* analyzer) {analyzer->setFlux(this);}
		~Flux(void){}
		virtual void update(std::vector<double>& flux, const std::vector<cellBoundary>& q_bdry) = 0;

		virtual double getMCS() const = 0;
		virtual std::string getName() const = 0;

	private :
        double max_characteristic_speed_;
		Solver* solver_;
};

#endif 
