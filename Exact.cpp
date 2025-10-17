#include "Exact.H"

void Exact::update(double* qe) {
    int gs = solver_->getGhostcell();
    int N_max = solver_->getNmax();
    double t = solver_->getT();
    const double* x = solver_->getX();
    double xl = solver_->getXL();
    double xr = solver_->getXR();
    double x_domain = xr - xl;


    if (idx_problem_ == 0) {
		auto g = [&](double x) -> double {
            return 1.0 + 0.5*sin(2.0*PI*(x-xl));
		};

        for (int i = 0; i < N_max; i++) {
            double x_new;
            double transfer = ADV_S*t;
            if (transfer > 0) {
                x_new = fmod(-fabs(x[i] - xr) - transfer, x_domain) + xr;
            } else {
                x_new = fmod(fabs(x[i] - xl) - transfer, x_domain) + xl;
            }
            qe[i] = g(x_new);
        }

    } else if (idx_problem_ == 1) {
		auto g = [&](double x) -> double {
            double xc = (xr + xl)*0.5;
            if (xc-0.1 <= x && x <= xc+0.1) {
                return 1.0;
            } else {
                return 0.1;
            }
		};

        for (int i = 0; i < N_max; i++) {
            double x_new;
            double transfer = ADV_S*t;
            if (transfer > 0) {
                x_new = fmod(-fabs(x[i] - xr) - transfer, x_domain) + xr;
            } else {
                x_new = fmod(fabs(x[i] - xl) - transfer, x_domain) + xl;
            }
            qe[i] = g(x_new);
        }
    } else if (idx_problem_ == 2) {
		auto g = [&](double x) -> double {
            double a = 0.5,
                   z = -0.7,
                   delta = 0.005,
                   alpha = 10.0,
                   beta = log(2.0)/(36.0*delta*delta);
        	auto G = [&](double x, double beta, double z) -> double {
			    return exp(-beta*(x-z)*(x-z));
		    };
		    auto F = [&](double x, double alpha, double a) -> double {
			    return sqrt(std::max(1.0 - alpha*alpha*(x-a)*(x-a), 0.0));
		    };

			if (-0.8 <= x && x <= -0.6) {
				return (G(x,beta,z-delta) + G(x,beta,z+delta) + 4.0*G(x,beta,z))/6.0;
			} else if (-0.4 <= x && x <= -0.2) {
				return 1.0;
			} else if (0.0 <= x && x <= 0.2) {
				return 1.0 - fabs(10.0*(x-0.1));
			} else if (0.4 <= x && x <= 0.6) {
				return (F(x,alpha,a-delta) + G(x,alpha,a+delta) + 4.0*F(x,alpha,a))/6.0;
			} else {
				return 0.0;
			}
		};
        for (int i = 0; i < N_max; i++) {
            double x_new;
            double transfer = ADV_S*t;
            if (transfer > 0) {
                x_new = fmod(-fabs(x[i] - xr) - transfer, x_domain) + xr;
            } else {
                x_new = fmod(fabs(x[i] - xl) - transfer, x_domain) + xl;
            }
            qe[i] = g(x_new);
        }
    } else if (idx_problem_ == 7) {
		auto g = [&](double x) -> double {
            if (-1.0 <= x && x <= 0.0) {
                return -sin(PI*x) - 0.5*x*x*x;
            } else if (0.0 < x && x <= 1.0) {
                return -sin(PI*x) - 0.5*x*x*x + 1.0;
            } else {
                return 0.0;
            }
		};


        for (int i = 0; i < N_max; i++) {
            double x_new;
            double transfer = ADV_S*t;
            if (transfer > 0) {
                x_new = fmod(-fabs(x[i] - xr) - transfer, x_domain) + xr;
            } else {
                x_new = fmod(fabs(x[i] - xl) - transfer, x_domain) + xl;
            }
            qe[i] = g(x_new);
        }
    } else if (idx_problem_ == 8) {
		auto g = [&](double x) -> double {
            return pow(sin(PI*x), 3);
		};


        for (int i = 0; i < N_max; i++) {
            double x_new;
            double transfer = ADV_S*t;
            if (transfer > 0) {
                x_new = fmod(-fabs(x[i] - xr) - transfer, x_domain) + xr;
            } else {
                x_new = fmod(fabs(x[i] - xl) - transfer, x_domain) + xl;
            }
            qe[i] = g(x_new);
        }
    }
}

void Exact::setProblem(int idx_problem) {
    idx_problem_ = idx_problem;
}

void Exact::initialize(double* qe) {}

