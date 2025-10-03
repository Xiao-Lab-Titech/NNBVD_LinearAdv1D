#ifndef _MUSCL_H_
#define _MUSCL_H_ 
#include <chrono>
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"
#include "../RawVector.H"

enum limiter {minmod, superbee, vanLeer, vanAlbada};

class MUSCL : public Reconstruction 
{
	public :
		MUSCL(Solver* slv, limiter lim = limiter::vanLeer);
		~MUSCL(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "MUSCL";return s;}
		double getReconstructionTime() const {return recon_time_;}

	private :
		double calcMUSCL(double qm1, double q0, double qp1) const;
		double minmod(double r) const;
		double superbee(double r) const;
		double vanLeer(double r) const;
		double vanAlbada(double r) const;
		double sign(double x) const;

	private :
		Solver* solver_;
		limiter lim_;

		double recon_time_;
		int N_stencil_ = 3;
	
		double** q_tmp_;
};


MUSCL::MUSCL(Solver* slv, limiter lim) : solver_(slv), lim_(lim) {
	solver_->setReconstruction(this);
	recon_time_ = 0.0;

	int N_max = solver_->getNmax();
	initVector2d<double>(q_tmp_, 3, N_max, 0.0);
}

MUSCL::~MUSCL(void) {
	freeVector2d<double>(q_tmp_, 3);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void MUSCL::update(cellBoundary** q_bdry, const double* const* q) {
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
			q_bdry[j][i+1].L = calcMUSCL(stencil[j][0], stencil[j][1], stencil[j][2]);
			q_bdry[j][i+1].R = calcMUSCL(stencil[j][3], stencil[j][2], stencil[j][1]);

		}
	}

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		if (RECONSTRUCTION_VAR == 0 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2consX(q_bdry, q_tmp_, gamma, i-1, i, i);
		if (RECONSTRUCTION_VAR == 1 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2primX(q_bdry, q_tmp_, gamma, i-1, i, i);
	}

	freeVector2d<double>(stencil, 3);
}

double MUSCL::calcMUSCL(double qm1, double q0, double qp1) const {
	double qL;
	double phi_L, r_L;
	
	r_L = (q0 - qm1)/(qp1 - q0 + sign(qp1 - q0)*1.0e-5);
	
	if(lim_ == limiter::minmod) {
		phi_L = minmod(r_L);
	} else if(lim_ == limiter::superbee) {
		phi_L = superbee(r_L);
	} else if(lim_ == limiter::vanAlbada) {
		phi_L = vanAlbada(r_L);
	} else if(lim_ == limiter::vanLeer) {
		phi_L = vanLeer(r_L);
	}
	//phi_L = 0.0;
	qL = q0 + 0.5*phi_L*(qp1 - q0);

	return qL;
}

double MUSCL::minmod(double r) const {
	return std::max(0.0, std::min(1.0, r));
}

double MUSCL::superbee(double r) const {
	return std::max(0.0, std::max(std::min(1.0, 2.0*r), std::min(2.0, r)));
}

double MUSCL::vanLeer(double r) const {
	double abs_r = std::abs(r);
	return (r + abs_r)/(1.0 + abs_r);
}

double MUSCL::vanAlbada(double r) const {
	double rr = r*r;
	return (r + rr)/(1.0 + rr);
}

double MUSCL::sign(double x) const {
	if(x >= 0.0) return 1.0;
	else return -1.0;
}

#endif 