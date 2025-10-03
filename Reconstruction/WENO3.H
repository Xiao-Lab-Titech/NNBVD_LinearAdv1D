#ifndef _WENO3_H_
#define _WENO3_H_ 

#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"


class WENO3 : public Reconstruction 
{
	public :
		WENO3(Solver* slv, std::string m = "Z");
		~WENO3(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "WENO3";return s;}
		double getReconstructionTime() const {return 0;}

	private :
		double calcWENO3(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;
		std::string minor_;
		const int N_stencil_ = 3;
		double** q_tmp_;
		
};


WENO3::WENO3(Solver* slv, std::string m) : solver_(slv), minor_(m) {
	solver_->setReconstruction(this);
	int N_max = solver_->getNmax();
	initVector2d(q_tmp_, 3, N_max, 0.0);

}

WENO3::~WENO3(void) {
	freeVector2d(q_tmp_, 3);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void WENO3::update(cellBoundary** q_bdry, const double* const* q) {
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
			q_bdry[j][i+1].L = calcWENO3(stencil[j][0], stencil[j][1], stencil[j][2]);
			q_bdry[j][i+1].R = calcWENO3(stencil[j][3], stencil[j][2], stencil[j][1]);

		}
	}

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		if (RECONSTRUCTION_VAR == 0 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2consX(q_bdry, q_tmp_, gamma, i-1, i, i);
		if (RECONSTRUCTION_VAR == 1 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2primX(q_bdry, q_tmp_, gamma, i-1, i, i);
	}

	freeVector2d<double>(stencil, 3);

}



double WENO3::calcWENO3(double qm1, double q0, double qp1) const {
	double qL; // L value is to i+1/2
	double ul0, ul1, w0, w1, alp0, alp1, beta0, beta1;
	double global_smooth_indicator, eps;
	double ideal_weight0 = 1.0/3.0, ideal_weight1 = 2.0/3.0;
	ul0 = (3.0*q0 - qm1)*0.5;
	ul1 = (q0 + qp1)*0.5;

	beta0 = (q0-qm1)*(q0-qm1);
	beta1 = (q0-qp1)*(q0-qp1);

	global_smooth_indicator = fabs(beta0-beta1);
	eps = 1.0e-15;
	alp0 = ideal_weight0*(1.0 + global_smooth_indicator/(beta0+eps));
	alp1 = ideal_weight1*(1.0 + global_smooth_indicator/(beta1+eps));

	w0 = alp0/(alp0+alp1);
	w1 = alp1/(alp0+alp1);

	qL =  w0*ul0 + w1*ul1;
	return qL;
}

#endif 