#include "Exact.H"

void Exact::update(double* qe) {
    int gs = solver_->getGhostcell();
    int N_max = solver_->getNmax();
    double t = solver_->getT();
    const double* x = solver_->getX();
    double xl = solver_->getXL();
    double xr = solver_->getXR();
    double x_domain = xr - xl;

    // Require problem to provide exact function. If not set, do nothing.
    if (!g_func_) return;

    for (int i = 0; i < N_max; i++) {
        double x_new;
        double transfer = ADV_S*t;
        if (transfer > 0) {
            x_new = fmod(-fabs(x[i] - xr) - transfer, x_domain) + xr;
        } else {
            x_new = fmod(fabs(x[i] - xl) - transfer, x_domain) + xl;
        }
        qe[i] = g_func_(x_new);
    }
}

void Exact::setProblem(int idx_problem) {
    idx_problem_ = idx_problem;
}

void Exact::initialize(double* qe) {}
