#ifndef _W3TBVD_H_
#define _W3TBVD_H_ 
#include <chrono>
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"
#include "../RawVector.H"

class W3TBVD : public Reconstruction 
{
	public :
		W3TBVD(Solver* slv);
		~W3TBVD(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "W3TBVD";return s;}
		double getReconstructionTime() const {return recon_time_;}
		double getTransformTime() const {return transfrom_time_;}
		double getreconstructionTime() const {return reconstruction_time_;}
		double getTBVTime() const {return TBV_time_;}
		double getSelectionTime() const {return selection_time_;}

	private :
		double calcTHINC(double qm1, double q0, double qp1) const;
		double calcWENO3(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;

		double recon_time_;
		double reconstruction_time_;
		double transfrom_time_;
		double TBV_time_;
		double selection_time_;
	

		double beta_;
		double T1_;
		const int N_stencil_ = 3;
		cellBoundary** q_bdry_tmp_;
		double*** TBV_;
		double** q_tmp_;
};


W3TBVD::W3TBVD(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
	recon_time_ = 0.0;
	reconstruction_time_ = 0.0;
	transfrom_time_ = 0.0;
	TBV_time_ = 0.0;
	selection_time_ = 0.0;

	beta_ = 1.6;
	T1_ = tanh(0.5*beta_);

	int N_max = solver_->getNmax();
	initVector2d<cellBoundary>(q_bdry_tmp_, 3, N_max, {0.0, 0.0});
	initVector3d<double>(TBV_, 2, 3, N_max, 0.0);
	initVector2d<double>(q_tmp_, 3, N_max, 0.0);
}

W3TBVD::~W3TBVD(void) {
	//int N_max = solver_->getNmax();
	freeVector2d<cellBoundary>(q_bdry_tmp_, 3);
	freeVector3d<double>(TBV_, 2, 3);
	freeVector2d<double>(q_tmp_, 3);
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void W3TBVD::update(cellBoundary** q_bdry, const double* const* q) {
	const int N_max = solver_->getNmax();
	const int gs = solver_->getGhostcell();
	const double gamma = solver_->getGamma();
	double** stencil;
	initVector2d<double>(stencil, 3, N_stencil_ + 1, 0.0);

	double start = omp_get_wtime();
	Reconstruction::transform(RECONSTRUCTION_VAR, solver_, q_tmp_, q);
	double time_stamp1 = omp_get_wtime();

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

			q_bdry_tmp_[j][i+1].L = calcTHINC(stencil[j][0], stencil[j][1], stencil[j][2]);
			q_bdry_tmp_[j][i+1].R = calcTHINC(stencil[j][3], stencil[j][2], stencil[j][1]);
		}
	}

	double time_stamp2 = omp_get_wtime();
	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			TBV_[0][i][j] = fabs(q_bdry[i][j].L - q_bdry[i][j].R) + fabs(q_bdry[i][j+1].L - q_bdry[i][j+1].R);
			TBV_[1][i][j] = fabs(q_bdry_tmp_[i][j].L - q_bdry_tmp_[i][j].R) + fabs(q_bdry_tmp_[i][j+1].L - q_bdry_tmp_[i][j+1].R);
		}
	}

	double time_stamp3 = omp_get_wtime();
	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			if (TBV_[0][i][j] > TBV_[1][i][j]) {
				q_bdry[i][j+1].L = q_bdry_tmp_[i][j+1].L;
				q_bdry[i][j].R = q_bdry_tmp_[i][j].R;
				solver_->setSelector(i, j, 1);
			} else {
				solver_->setSelector(i, j, 0);
			}
		}
	}

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		if (RECONSTRUCTION_VAR == 0 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2consX(q_bdry, q_tmp_, gamma, i-1, i, i);
		if (RECONSTRUCTION_VAR == 1 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2primX(q_bdry, q_tmp_, gamma, i-1, i, i);
	}

	freeVector2d<double>(stencil, 3);

	double end = omp_get_wtime();
	//end = std::chrono::system_clock::now(); // 計測終了
	//double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	double elapsed = end - start;
	//printf("recon time: %f\n", elapsed);
	transfrom_time_ += time_stamp1 - start;
	reconstruction_time_ += time_stamp2 - time_stamp1;
	TBV_time_ += time_stamp3 - time_stamp2;
	selection_time_ += end - time_stamp3;
	recon_time_ += elapsed;
}



double W3TBVD::calcTHINC(double qm1, double q0, double qp1) const {
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

double W3TBVD::calcWENO3(double qm1, double q0, double qp1) const {
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

	qL = w0*ul0 + w1*ul1;
	return qL;
}

#endif 