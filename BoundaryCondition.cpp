#include "BoundaryCondition.H"

void BoundaryCondition::update(std::vector<double>& q) {
    int gs = solver_->getGhostcell();
    int N_max = solver_->getNmax();
    // If user provided a custom left-side boundary function, use it.
    if (bc_func_l_) {
        bc_func_l_(q, solver_);
    } else {
        switch (bc_l_) {
            case open:
                for (int i = 0; i < gs; i++){
                    q[i] = q[gs];
                }
                break;
            case periodic:
                for (int i = 0; i < gs; i++){
                    q[i] = q[N_max - 2*gs + i];
                }
                break;
            case reflective_wall:
                for (int i = 0; i < gs; i++){
                    q[i] = -q[2*gs - 1 - i];
                }
                break;
            default :
                // debugging
                break;
        }
    }

        // If user provided a custom right-side boundary function, use it.
        if (bc_func_r_) {
            bc_func_r_(q, solver_);
        } else {
            switch (bc_r_) {
                case open:
                    for (int i = 0; i < gs; i++){
                        q[N_max - 1 - i] = q[N_max - gs - 1];
                    }
                    break;
                case periodic:
                    for (int i = 0; i < gs; i++){
                        q[N_max - 1 - i] = q[2*gs - i - 1];
                    }
                    break;
                case reflective_wall:
                    for (int i = 0; i < gs; i++){
                        q[N_max - 1 - i] = q[N_max - 2*gs + i];
                    }
                    break;
                default :
                    // debugging
                    break;
            }
        }
}