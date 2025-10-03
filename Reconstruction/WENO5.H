#ifndef _WENO5_H_
#define _WENO5_H_ 
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"


class WENO5 : public Reconstruction 
{
	public :
		WENO5(Solver* slv, std::string m = "Z");
		~WENO5(void);
		void update(cellBoundary* q_bdry, const double* q);
		std::string getName() const {std::string s = "WENO5";return s;}
		double getReconstructionTime() const {return recon_time_;}

	private :
		double calcWENO5(std::string m, double qm2, double qm1, double q0, double qp1, double qp2) const;

	private :
		Solver* solver_;
		std::string minor_;

		const int N_stencil_ = 5;
		double recon_time_;
		
		//double lambda_Zplus;
};


WENO5::WENO5(Solver* slv, std::string m) : solver_(slv), minor_(m) {
	solver_->setReconstruction(this);
	int N_max = solver_->getNmax();
	recon_time_ = 0.0;
	//lambda_Zplus = pow(solver_->getDX(), 2.0/3.0);
}

WENO5::~WENO5(void) {
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void WENO5::update(cellBoundary* q_bdry, const double* q) {
	const int N_max = solver_->getNmax();
	const int gs = solver_->getGhostcell();

	double start = omp_get_wtime();
	for (int i = gs - 1; i < N_max - gs; i++) {
		q_bdry[i+1].L = calcWENO5(minor_, q[i-2], q[i-1], q[i], q[i+1], q[i+2]);
		q_bdry[i+1].R = calcWENO5(minor_, q[i+3], q[i+2], q[i+1], q[i], q[i-1]);
	}
	double end = omp_get_wtime();
	double elapsed = end - start;
	recon_time_ += elapsed;
}



double WENO5::calcWENO5(std::string m, double qm2, double qm1, double q0, double qp1, double qp2) const {
	double qL; // L value is to i+1/2
	double ul0, ul1, ul2, w0, w1, w2, alp0, alp1, alp2, beta0, beta1, beta2;
	double gamma0, gamma1, u00, u01, u02, u10, u11, u12; // for WENO-ZA
	double eps = 1.0e-15, tmp0, tmp1;
	double ideal_weight0 = 0.1, ideal_weight1 = 0.6, ideal_weight2 = 0.3;
	ul0 = (2.0*qm2 - 7.0*qm1 + 11.0*q0)/6.0;
	ul1 = (-qm1 + 5.0*q0 + 2.0*qp1)/6.0;
	ul2 = (2.0*q0 + 5.0*qp1 - qp2)/6.0;

	if (m == "eta") {
		tmp0 = qm2 - 2.0*qm1 + q0;
		tmp1 = qm2 - 4.0*qm1 + 3.0*q0;
		beta0 = tmp0*tmp0 + tmp1*tmp1/4.0;

		tmp0 = qm1 - 2.0*q0 + qp1;
		tmp1 = qm1 - qp1;
		beta1 = tmp0*tmp0 + tmp1*tmp1/4.0;

		tmp0 = q0 - 2.0*qp1 + qp2;
		tmp1 = 3.0*q0 - 4.0*qp1 + qp2;
		beta2 = tmp0*tmp0 + tmp1*tmp1/4.0;
	} else if (m == "ZA") {
		gamma0 = 1.0;
		gamma1 = 13.0/12.0;

		u10 = qm2 - 2.0*qm1 + q0;
		u00 = (qm2 - 4.0*qm1 + 3.0*q0)*0.5;
		beta0 = gamma0*u00*u00 + gamma1*u10*u10;

		u11 = qm1 - 2.0*q0 + qp1;
		u01 = -(qm1 - qp1)*0.5;
		beta1 = gamma0*u01*u01 + gamma1*u11*u11;

		u12 = q0 - 2.0*qp1 + qp2;
		u02 = -(3.0*q0 - 4.0*qp1 + qp2)*0.5;
		beta2 = gamma0*u02*u02 + gamma1*u12*u12;	
	} else if (m == "NIP") {
		double theta = 0.1;
		beta0 = theta*abs(qm2 - 3.0*qm1 + 2.0*q0) + abs(qm2 - 2.0*qm1 + q0);
		beta1 = theta*abs(qp1 - q0) + abs(qm1 - 2.0*q0 + qp1);
		beta2 = theta*abs(qp1 - q0) + abs(q0 - 2.0*qp1 + qp2);
	} else {
		tmp0 = qm2 - 2.0*qm1 + q0;
		tmp1 = qm2 - 4.0*qm1 + 3.0*q0;
		beta0 = 13.0/12.0*tmp0*tmp0 + tmp1*tmp1/4.0;

		tmp0 = qm1 - 2.0*q0 + qp1;
		tmp1 = qm1 - qp1;
		beta1 = 13.0/12.0*tmp0*tmp0 + tmp1*tmp1/4.0;

		tmp0 = q0 - 2.0*qp1 + qp2;
		tmp1 = 3.0*q0 - 4.0*qp1 + qp2;
		beta2 = 13.0/12.0*tmp0*tmp0 + tmp1*tmp1/4.0;	
	}


	if (m == "ENO") {
		alp0 = 0.0; alp1 = 0.0; alp2 = 0.0;
		double B[3] = {beta0, beta1, beta2};
		double min_beta = B[0];
		int idx_min = 0;
		for (int i = 0; i < 3; i++) {
			if (B[i] < min_beta) {
				min_beta = B[i];
				idx_min = i;
			}
		}
		if (idx_min == 0) {
			alp0 = 1.0;
		} else if (idx_min == 1) {
			alp1 = 1.0;
		} else if (idx_min == 2) {
			alp2 = 1.0;
		}
	} else if (m == "JS") {
		alp0 = ideal_weight0/((beta0+eps)*(beta0+eps));
		alp1 = ideal_weight1/((beta1+eps)*(beta1+eps));
		alp2 = ideal_weight2/((beta2+eps)*(beta2+eps));
	} else if (m == "Z" || m == "eta") {
		double gsi = fabs(beta0-beta2);
		tmp0 = gsi/(beta0+eps);
		alp0 = ideal_weight0*(1.0 + tmp0*tmp0);
		tmp0 = gsi/(beta1+eps);
		alp1 = ideal_weight1*(1.0 + tmp0*tmp0);
		tmp0 = gsi/(beta2+eps);
		alp2 = ideal_weight2*(1.0 + tmp0*tmp0);
	} else if (m == "Zplus") {
		double gsi = fabs(beta0-beta2);
		double lambda_Zplus = pow(solver_->getDX(), 2.0/3.0);
		tmp0 = (gsi+eps)/(beta0+eps);
		alp0 = ideal_weight0*(1.0 + tmp0*tmp0 + lambda_Zplus/tmp0);
		tmp0 = (gsi+eps)/(beta1+eps);
		alp1 = ideal_weight1*(1.0 + tmp0*tmp0 + lambda_Zplus/tmp0);
		tmp0 = (gsi+eps)/(beta2+eps);
		alp2 = ideal_weight2*(1.0 + tmp0*tmp0 + lambda_Zplus/tmp0);
	} else if (m == "ZA") {
		tmp0 = abs(u00)-abs(u02);
		tmp1 = abs(u10)-abs(u12);
		double gsi6 = gamma0*tmp0*tmp0 + gamma1*tmp1*tmp1;
		double A = gsi6/(beta0+beta2-gsi6+eps);
		alp0 = ideal_weight0*(1.0 + A*gsi6/(beta0+eps));
		alp1 = ideal_weight1*(1.0 + A*gsi6/(beta1+eps));
		alp2 = ideal_weight2*(1.0 + A*gsi6/(beta2+eps));
	} else if (m == "A") {
		double phi = std::min(1.0, std::sqrt(abs(beta0-2.0*beta1+beta2)));
		double gsi = fabs(beta0-beta2);
		tmp0 = gsi/(beta0+eps);
		alp0 = ideal_weight0*(std::max(1.0, phi*(tmp0*tmp0)));
		tmp0 = gsi/(beta1+eps);
		alp1 = ideal_weight1*(std::max(1.0, phi*(tmp0*tmp0)));
		tmp0 = gsi/(beta2+eps);
		alp2 = ideal_weight2*(std::max(1.0, phi*(tmp0*tmp0)));
	} else if (m == "NIP") {
		tmp0 = qm2-4.0*qm1+6.0*q0-4.0*qp1+qp2;
		double gsi = tmp0*tmp0; // J = 2 in my code
		alp0 = ideal_weight0*(1.0 + gsi/((beta0+eps)*(beta0+eps)));
		alp1 = ideal_weight1*(1.0 + gsi/((beta1+eps)*(beta1+eps)));
		alp2 = ideal_weight2*(1.0 + gsi/((beta2+eps)*(beta2+eps)));
	} else {
		alp0 = ideal_weight0/((beta0+eps)*(beta0+eps));
		alp1 = ideal_weight1/((beta1+eps)*(beta1+eps));
		alp2 = ideal_weight2/((beta2+eps)*(beta2+eps));
	}

	w0 = alp0/(alp0+alp1+alp2);
	w1 = alp1/(alp0+alp1+alp2);
	w2 = alp2/(alp0+alp1+alp2);

	qL =  w0*ul0 + w1*ul1 + w2*ul2;
	return qL;
}

#endif 