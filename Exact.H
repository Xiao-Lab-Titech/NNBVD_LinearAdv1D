#ifndef _EXACT_H_
#define _EXACT_H_
#include "Solver.H"
#include "Parameter.H"
// Exact solution of 1D Riemann problem for general initial values

// Reference: E.F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009).

struct Variable {
	double L;
	double R;
};


class Exact {
	public :
        Exact(Solver* slv) : solver_(slv) {}
		~Exact(){}

		void update(double* qe);
        void setProblem(int idx_problem);
        void initialize(double* qe);

    private :

    private :
        Solver* solver_;
        int idx_problem_;

};


#endif
