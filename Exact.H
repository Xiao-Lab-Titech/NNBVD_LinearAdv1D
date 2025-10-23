#ifndef _EXACT_H_
#define _EXACT_H_
#include "Solver.H"
#include "Parameter.H"
//#include <functional>
//#include <vector>
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

        // function signature: given x return exact solution value
        using ExactFunc = std::function<double(double)>;

        void update(std::vector<double>& qe);
        void setProblem(int idx_problem);
        void setExactFunction(ExactFunc f) { g_func_ = f; }
        void initialize(std::vector<double>& qe);

    private :
        Solver* solver_;
        int idx_problem_;
        ExactFunc g_func_;

};


#endif
