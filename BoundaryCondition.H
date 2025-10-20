#ifndef _BOUNDARYCONDITION_H_
#define _BOUNDARYCONDITION_H_
#include "Solver.H"
#include <functional>

//class Solver;

enum bdry {open, periodic, reflective_wall};

class BoundaryCondition {
    public :
        // callback signature: function that fills ghost cells on a boundary.
        // Parameters: pointer to solution array q, pointer to Solver for geometry/info.
        using BCFunc = std::function<void(double* /*q*/, Solver* /*solver*/)>
            ;

        BoundaryCondition(Solver* slv) : solver_(slv) {}
        ~BoundaryCondition(){}

        void update(double* q);
        // setBC keeps backward compatibility. You can optionally pass two callbacks
        // (left and right). If a callback is non-empty it will be used instead of
        // the built-in enum behaviors for that side.
        void setBC(int l, int r, BCFunc func_l = BCFunc(), BCFunc func_r = BCFunc()) {
            bc_l_ = l; bc_r_ = r; bc_func_l_ = func_l; bc_func_r_ = func_r;
        }
        void getInfo();

    private :
        Solver* solver_;
        int bc_l_;
        int bc_r_;
        BCFunc bc_func_l_;
        BCFunc bc_func_r_;
};




#endif
