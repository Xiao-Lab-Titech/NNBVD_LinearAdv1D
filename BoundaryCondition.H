#ifndef _BOUNDARYCONDITION_H_
#define _BOUNDARYCONDITION_H_
#include "Solver.H"

//class Solver;

enum bdry {open, periodic, reflective_wall};

class BoundaryCondition {
	public :
        BoundaryCondition(Solver* slv) : solver_(slv) {}
		~BoundaryCondition(){}

		void update(double* q);
        void setBC(int l, int r) {bc_l_ = l; bc_r_ = r;}
        void getInfo();

    private :
        Solver* solver_;
        int bc_l_;
        int bc_r_;
};




#endif
