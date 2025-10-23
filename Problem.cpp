#include "Problem.H"
#include "Parameter.H"

void LinearAdv1DProb::initProblem(std::vector<double>& q, std::vector<double>& qe) {
    if (idx_ < 0 || idx_ >= static_cast<int>(init_table_.size())) {
        throw std::out_of_range("Unknown problem index: " + std::to_string(idx_));
    }
    auto memfn = init_table_[idx_];
    if (memfn == nullptr) {
        throw std::runtime_error("Uninitialized init_table_ entry at index: " + std::to_string(idx_));
    }
    (this->*memfn)(q, qe);
}

void LinearAdv1DProb::sineWave(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x = solver_->getX();

    name_ = "Sine Wave";
    has_exact_ = true;
    xl_ = 0.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);
    q.resize(N_max);
    qe.resize(N_max);
    for (int i = 0; i < N_max; i++) {
        q[i] = 1.0 + 0.5*sin(2.0*PI*(x[i]-x[gs]));
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    // Default enum usage (unchanged):
    bc_->setBC(periodic, periodic);

    // Example: set custom boundary functions for this problem.
    // Uncomment and modify the lambdas below to provide per-problem boundary logic.
    /*
    bc_->setBC(periodic, periodic,
        // left boundary lambda: fill left ghost cells
        [](double* q, Solver* solver){
            int gs = solver->getGhostcell();
            int N_max = solver->getNmax();
            const double* x = solver->getX();
            // example: linear extrapolation from first interior cell
            for (int i = 0; i < gs; ++i) {
                q[i] = q[gs] + (q[gs] - q[gs+1]) * (gs - i);
            }
        },
        // right boundary lambda: set to zero (Dirichlet)
        [](double* q, Solver* solver){
            int gs = solver->getGhostcell();
            int N_max = solver->getNmax();
            for (int i = 0; i < gs; ++i) {
                q[N_max - 1 - i] = 0.0;
            }
        }
    );
    */
    // You can supply the exact solution as a function from Problem to Exact:
    // Example for sine wave:
    // supply exact function (function-based exact, no fallback)
    exact_->setExactFunction([xl = xl_](double x) -> double { return 1.0 + 0.5*sin(2.0*PI*(x-xl)); });
    //exact_->initialize(qe_);

}

void LinearAdv1DProb::squareWave(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x = solver_->getX();

    name_ = "Square Wave";
    has_exact_ = true;
    xl_ = 0.0; xr_ = 1.0;
    double xc = (xr_+xl_)*0.5;
    solver_->initCell(xl_, xr_);
    q.resize(N_max);
    qe.resize(N_max);
    for (int i = 0; i < N_max; i++) {
        if (xc-0.1 <= x[i] && x[i] <= xc+0.1) {
            q[i] = 1.0;
        } else {
            q[i] = 0.1;
        }
        qe[i] = q[i];
    }
    te_ = 1.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(periodic, periodic);
    exact_->setExactFunction([xl = xl_, xr = xr_](double x) -> double {
        double xc = (xr + xl)*0.5;
        if (xc-0.1 <= x && x <= xc+0.1) return 1.0; else return 0.1;
    });
    //exact_->initialize(qe_);
}

void LinearAdv1DProb::JiangAndShu(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x = solver_->getX();

    name_ = "Jiang and Shu";
    has_exact_ = true;
    xl_ = -1.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);
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
            q[i] = (G(x[i],beta,z-delta) + G(x[i],beta,z+delta) + 4.0*G(x[i],beta,z))/6.0;
        } else if (-0.4 <= x[i] && x[i] <= -0.2) {
            q[i] = 1.0;
        } else if (0.0 <= x[i] && x[i] <= 0.2) {
            q[i] = 1.0 - fabs(10.0*(x[i]-0.1));
        } else if (0.4 <= x[i] && x[i] <= 0.6) {
            q[i] = (F(x[i],alpha,a-delta) + G(x[i],alpha,a+delta) + 4.0*F(x[i],alpha,a))/6.0;
        } else {
            q[i] = 0.0;
        }
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(periodic, periodic);
    exact_->setExactFunction([a = 0.5, z = -0.7]() -> std::function<double(double)> {
        return [a, z](double x) -> double {
            double delta = 0.005;
            double alpha = 10.0;
            double beta = log(2.0)/(36.0*delta*delta);
            auto G = [&](double x, double beta, double z) -> double { return exp(-beta*(x-z)*(x-z)); };
            auto F = [&](double x, double alpha, double a) -> double { return sqrt(std::max(1.0 - alpha*alpha*(x-a)*(x-a), 0.0)); };
            if (-0.8 <= x && x <= -0.6) return (G(x,beta,z-delta) + G(x,beta,z+delta) + 4.0*G(x,beta,z))/6.0;
            else if (-0.4 <= x && x <= -0.2) return 1.0;
            else if (0.0 <= x && x <= 0.2) return 1.0 - fabs(10.0*(x-0.1));
            else if (0.4 <= x && x <= 0.6) return (F(x,alpha,a-delta) + G(x,alpha,a+delta) + 4.0*F(x,alpha,a))/6.0;
            else return 0.0;
        };
    }());
    //exact_->initialize(qe_);
}

void LinearAdv1DProb::randomSine(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x = solver_->getX();

    name_ = "Random Sine";
    has_exact_ = false;
    xl_ = 0.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);

    q.resize(N_max);
    qe.resize(N_max);

    unsigned int seed = 123;
    //std::mt19937 mt{ seed };
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution<double> dist1(1.e-2, 50);
    std::uniform_int_distribution<int> dist2(1, 10);
    double A = dist1(mt);
    double f = dist2(mt);

    for (int i = 0; i < N_max; i++) {
        q[i] = A*sin(2.0*PI*f*(x[i]-x[gs]));
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(periodic, periodic);
    //exact_->setProblem(idx_);
    //exact_->initialize(qe_);
}
void LinearAdv1DProb::randomSquare(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x = solver_->getX();

    name_ = "Random Square";
    has_exact_ = false;
    xl_ = 0.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);

    q.resize(N_max);
    qe.resize(N_max);

    unsigned int seed = 123;
    //std::mt19937 mt{ seed };
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution<double> dist(-50.0, 50.0);
    double xc = (xr_+xl_)*0.5;
    double A = dist(mt);

    for (int i = 0; i < N_max; i++) {
        if (xc-0.1 <= x[i] && x[i] <= xc+0.1) {
            q[i] = A;
        } else {
            q[i] = 0.0;
        }
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(periodic, periodic);
    //exact_->setProblem(idx_);
    //exact_->initialize(qe_);

}
void LinearAdv1DProb::randomTriangle(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x = solver_->getX();

    name_ = "Random Triangle";
    has_exact_ = false;
    xl_ = 0.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);

    unsigned int seed = 123;
    //std::mt19937 mt{ seed };
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution<double> dist1(-50.0,50.0);
    std::uniform_int_distribution<int> dist2(1, 8);
    double f = 2*dist2(mt);
    double xc = (xr_+xl_)/f/2;
    double top = dist1(mt);
    double hh = xc;
    int j = 1;
    for (int i = 0; i < N_max; i++) {
        j = 2*int(x[i]/((xr_-xl_)/f)) + 1;
        if ((j-1) % 4 == 0) {
            q[i] = top*(1.0 - fabs(x[i]-j*xc)/hh);
        } else {
            q[i] = -top*(1.0 - fabs(x[i]-j*xc)/hh);

        }
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(periodic, periodic);
    //exact_->setProblem(idx_);
    //exact_->initialize(qe_);
}
void LinearAdv1DProb::randomPoly(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x = solver_->getX();

    name_ = "Random Polynomial";
    has_exact_ = false;
    xl_ = 0.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);

    q.resize(N_max);
    qe.resize(N_max);

    unsigned int seed = 123;
    //std::mt19937 mt{ seed };
    std::mt19937 mt{ std::random_device{}() };
    std::uniform_real_distribution<double> dist0(-100.0,100.0);
    std::uniform_real_distribution<double> dist1(0.0,1.0);
    std::uniform_int_distribution<int> dist2(0, 1);
    double A = dist0(mt);
    double a[4];
    int b[4];
    for (int i = 0; i < 4; i++) {
        a[i] = dist1(mt);
        b[i] = dist2(mt);
    }
    for (int i = 0; i < N_max; i++) {
        q[i] = A;
        for (int j = 0; j < 3; j++) {
            if (b[j] == 1) {
                q[i] *= x[i]-a[j];
            }
        }
        q[i] *= x[i]-a[3];
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(open, open);
    //exact_->setProblem(idx_);
    //exact_->initialize(qe_);
}
void LinearAdv1DProb::nonlinearDiscontinuity(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x_vec = solver_->getX();

    name_ = "Nonlinear Discontinuity";
    has_exact_ = true;
    xl_ = -1.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);

    q.resize(N_max);
    qe.resize(N_max);
    for (int i = 0; i < N_max; i++) {
        double x = x_vec[i];
        if (x <= 0.0) {
            q[i] = -sin(PI*x) - 0.5*x*x*x;
        } else {
            q[i] = -sin(PI*x) - 0.5*x*x*x + 1.0;
        }
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(periodic, periodic);
    exact_->setExactFunction([xl = xl_](double x) -> double {
        if (x <= 0.0) return -sin(PI*x) - 0.5*x*x*x; else return -sin(PI*x) - 0.5*x*x*x + 1.0;
    });
    //exact_->initialize(qe_);
}
void LinearAdv1DProb::cubedSine(std::vector<double>& q, std::vector<double>& qe) {
    int N_max = solver_->getNmax();
    int gs = solver_->getGhostcell();
    const std::vector<double>& x_vec = solver_->getX();

    name_ = "Cubed Sine";
    has_exact_ = true;
    xl_ = -1.0; xr_ = 1.0;
    solver_->initCell(xl_, xr_);

    q.resize(N_max);
    qe.resize(N_max);

    for (int i = 0; i < N_max; i++) {
        double x = x_vec[i];
        q[i] = pow(sin(PI*x), 3);
        qe[i] = q[i];
    }
    te_ = 2.0;
    
    bc_ = new BoundaryCondition(solver_);
    exact_ = new Exact(solver_); // new LinearAdv1DEx(solver_);
    solver_->setBoundaryCondition(bc_);
    solver_->setExact(exact_);
    bc_->setBC(periodic, periodic);
    exact_->setExactFunction([](double x) -> double { return pow(sin(PI*x), 3); });
    //exact_->initialize(qe_);
}
