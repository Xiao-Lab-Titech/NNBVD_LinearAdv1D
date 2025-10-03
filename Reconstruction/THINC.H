#ifndef _THINC_H_
#define _THINC_H_ 

#include "../Reconstruction.H"
#include "../Solver.H"

class THINC : public Reconstruction 
{
	public :
		THINC(Solver* slv);
		~THINC(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "THINC";return s;}
		double getReconstructionTime() const {return 0;}


	private :
		double calcTHINC(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;
		const int N_stencil_ = 3;
		double** q_tmp_;
		
		double beta_;
		double T1_;
		
};


THINC::THINC(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
	int N_max = solver_->getNmax();

	beta_ = 1.6;
	T1_ = tanh(0.5*beta_);

	initVector2d(q_tmp_, 3, N_max, 0.0);
}

THINC::~THINC(void) {
	freeVector2d(q_tmp_, 3);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void THINC::update(cellBoundary** q_bdry, const double* const* q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();
	const double gamma = solver_->getGamma();
	double** stencil;
	initVector2d<double>(stencil, 3, N_stencil_ + 1, 0.0);
	
	Reconstruction::transform(RECONSTRUCTION_VAR, solver_, q_tmp_, q);

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		for (int j = 0; j < N_stencil_ + 1; j++) {
			stencil[0][j] = q_tmp_[0][i - (N_stencil_-1)/2 + j];
			stencil[1][j] = q_tmp_[1][i - (N_stencil_-1)/2 + j];
			stencil[2][j] = q_tmp_[2][i - (N_stencil_-1)/2 + j];
		}

		if (RECONSTRUCTION_VAR == 0 && IS_CHARACTERISTIC_DECOMP) Reconstruction::cons2charX(stencil, N_stencil_, q_tmp_, gamma, i, i+1);
		if (RECONSTRUCTION_VAR == 1 && IS_CHARACTERISTIC_DECOMP) Reconstruction::prim2charX(stencil, N_stencil_, q_tmp_, gamma, i, i+1);
		
		for (int j = 0; j < 3; j++) {
			q_bdry[j][i+1].L = calcTHINC(stencil[j][0], stencil[j][1], stencil[j][2]);
			q_bdry[j][i+1].R = calcTHINC(stencil[j][3], stencil[j][2], stencil[j][1]);

		}
	}

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		if (RECONSTRUCTION_VAR == 0 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2consX(q_bdry, q_tmp_, gamma, i-1, i, i);
		if (RECONSTRUCTION_VAR == 1 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2primX(q_bdry, q_tmp_, gamma, i-1, i, i);
	}

	freeVector2d<double>(stencil, 3);
}

double THINC::calcTHINC(double qm1, double q0, double qp1) const {
	double qL; // L value is to i+1/2
	double alpha,epsilon;
	double T2;
	epsilon = 1e-15;
	alpha = 0.0;

    // left-side cell boundary value
	if ((q0-qm1)*(qp1-q0) > epsilon){
		alpha = -(qp1-2.0*q0+qm1)/(qp1-qm1);
		T2 = tanh(0.5*beta_*alpha);
		qL = 0.5*(qp1+qm1)+0.5*(qp1-qm1)*(T1_+T2/T1_)/(1.0+T2);
	} else {
		qL = q0;
	}

	return qL;
}

#endif 