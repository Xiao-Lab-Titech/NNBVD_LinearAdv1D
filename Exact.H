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
        double getOutsideCenterPos(double x_l, double x_c, double x_r, double domain_l, double domain_r);

    private :
        Solver* solver_;
        int idx_problem_;

};

void Exact::update(double* qe) {
    int gs = solver_->getGhostcell();
    int N_max = solver_->getNmax();
    double t = solver_->getT();
    const double* x = solver_->getX();
    double xl = solver_->getXL();
    double xr = solver_->getXR();


    if (idx_problem_ == 0) {
        for (int i = 0; i < N_max; i++) {
            qe[i] = 1.0 + 0.5*sin(2.0*PI*((x[i] - x[gs]) - ADV_S*t));
        }
    } else if (idx_problem_ == 1) {
        double x_domain = xr-xl;
        double xc0 = (xr-xl)*0.5; 
        double sq_now = fmod(xc0+ADV_S*t, x_domain);
        double sq_out = getOutsideCenterPos(sq_now-0.1,sq_now,sq_now+0.1,xl,xr); 
        bool is_eq_sq_pos = (abs(sq_now - sq_out) < 1.e-5) ? true : false;

		for (int i = 0; i < N_max; i++) {
            if (sq_now-0.1 <= x[i] && x[i] <= sq_now+0.1) {
				qe[i] = 1.0;
			} else if (!is_eq_sq_pos && (sq_out-0.1 <= x[i] && x[i] <= sq_out+0.1)) {
				qe[i] = 1.0;
            } else {
                qe[i] = 0.1;
            }
		}
    } else if (idx_problem_ == 2) {
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

        double x_domain = xr-xl;

        double xc_now = fmod(ADV_S*t, x_domain);
        double sq_now = fmod(xc_now - 0.3, x_domain);
        double tri_now = fmod(xc_now + 0.1, x_domain);
        double a_now = fmod(xc_now + a, x_domain);
        double z_now = fmod(xc_now + z, x_domain);

        double sq_out = getOutsideCenterPos(sq_now-0.1,sq_now,sq_now+0.1,xl,xr); 
        double tri_out = getOutsideCenterPos(tri_now-0.1,tri_now,tri_now+0.1,xl,xr); 
        double a_out = getOutsideCenterPos(a_now-0.1,a_now,a_now+0.1,xl,xr); 
        double z_out = getOutsideCenterPos(z_now-0.1,z_now,z_now+0.1,xl,xr);
        
        bool is_eq_sq_pos = (abs(sq_now - sq_out) < 1.e-5) ? true : false;
        bool is_eq_tri_pos = (abs(tri_now - tri_out) < 1.e-5) ? true : false;
        bool is_eq_a_pos = (abs(a_now - a_out) < 1.e-5) ? true : false;
        bool is_eq_z_pos = (abs(z_now - z_out) < 1.e-5) ? true : false;

		for (int i = 0; i < N_max; i++) {
            if (z_now-0.1 <= x[i] && x[i] <= z_now+0.1) {
				qe[i] = (G(x[i],beta,z_now-delta) + G(x[i],beta,z_now+delta) + 4.0*G(x[i],beta,z_now))/6.0;
			} else if (!is_eq_z_pos && (z_out-0.1 <= x[i] && x[i] <= z_out+0.1)) {
				qe[i] = (G(x[i],beta,z_out-delta) + G(x[i],beta,z_out+delta) + 4.0*G(x[i],beta,z_out))/6.0;
			} else if (sq_now-0.1 <= x[i] && x[i] <= sq_now+0.1) {
				qe[i] = 1.0;
			} else if (!is_eq_sq_pos && (sq_out-0.1 <= x[i] && x[i] <= sq_out+0.1)) {
				qe[i] = 1.0;
			} else if (tri_now-0.1 <= x[i] && x[i] <= tri_now+0.1) {
				qe[i] = 1.0 - fabs(10.0*(x[i]-tri_now));
			} else if (!is_eq_tri_pos && (tri_out-0.1 <= x[i] && x[i] <= tri_out+0.1)) {
				qe[i] = 1.0 - fabs(10.0*(x[i]-tri_out));
			} else if (a_now-0.1 <= x[i] && x[i] <= a_now+0.1) {
				qe[i] = (F(x[i],alpha,a_now-delta) + G(x[i],alpha,a_now+delta) + 4.0*F(x[i],alpha,a_now))/6.0;
			} else if (!is_eq_a_pos && (a_out-0.1 <= x[i] && x[i] <= a_out+0.1)) {
				qe[i] = (F(x[i],alpha,a_out-delta) + G(x[i],alpha,a_out+delta) + 4.0*F(x[i],alpha,a_out))/6.0;
			} else {
                qe[i] = 0.0;
            }
		}
    }


}

void Exact::setProblem(int idx_problem) {
    idx_problem_ = idx_problem;
}

void Exact::initialize(double* qe) {
    int gs = solver_->getGhostcell();
    int N_max = solver_->getNmax();
    const double* x = solver_->getX();
    double xl = solver_->getXL();
    double xr = solver_->getXR();

    if (idx_problem_ == 0) {
        for (int i = 0; i < N_max; i++) {
            qe[i] = 1.0 + 0.5*sin(2.0*PI*(x[i]-x[gs]));
        }
    } else if (idx_problem_ == 1) {
		double xc = (xr-xl)*0.5;
		for (int i = 0; i < N_max; i++) {
			if (xc-0.1 <= x[i] && x[i] <= xc+0.1) {
				qe[i] = 1.0;
			} else {
				qe[i] = 0.1;
			}
		}
    } else if (idx_problem_ == 2) {
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
		for (int i = 0; i < N_max; i++) {
			if (-0.8 <= x[i] && x[i] <= -0.6) {
				qe[i] = (G(x[i],beta,z-delta) + G(x[i],beta,z+delta) + 4.0*G(x[i],beta,z))/6.0;
			} else if (-0.4 <= x[i] && x[i] <= -0.2) {
				qe[i] = 1.0;
			} else if (0.0 <= x[i] && x[i] <= 0.2) {
				qe[i] = 1.0 - fabs(10.0*(x[i]-0.1));
			} else if (0.4 <= x[i] && x[i] <= 0.6) {
				qe[i] = (F(x[i],alpha,a-delta) + G(x[i],alpha,a+delta) + 4.0*F(x[i],alpha,a))/6.0;
			} else {
				qe[i] = 0.0;
			}
		}
    }
}

double Exact::getOutsideCenterPos(double x_l, double x_c, double x_r, double domain_l, double domain_r) {
    double d_len = domain_r - domain_l;
    if (x_l < domain_l) {
        return x_c + d_len;
    } else if (x_r > domain_r) {
        return x_c - d_len;
    } else {
        return x_c;
    }
}


#endif
