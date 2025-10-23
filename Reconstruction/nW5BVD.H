#ifndef _nW5BVD_H_
#define _nW5BVD_H_ 
#include <chrono>
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"
//#include <vector>

class nW5BVD : public Reconstruction 
{
	public :
		nW5BVD(Solver* slv, std::vector<std::string> M, int TBV_type = 0, bool is_log_selection = false);
		~nW5BVD(void);
		void update(std::vector<cellBoundary>& q_bdry, const std::vector<double>& q);
		std::string getName() const {std::string s = "nW5BVD";return s;}

		double getReconstructionTime() const {return recon_time_;}
		double getTransformTime() const {return transfrom_time_;}
		double getreconstructionTime() const {return reconstruction_time_;}
		double getTBVTime() const {return TBV_time_;}
		double getSelectionTime() const {return selection_time_;}

		std::vector<std::string> getCandidateFunctionLabels() const {
			std::vector<std::string> m_cpy(M_.size());
			std::copy(M_.begin(), M_.end(), m_cpy.begin());
			return m_cpy;
		}

	private :
		double calcWENO5(std::string m, double qm2, double qm1, double q0, double qp1, double qp2) const;
		double calcTHINC(double qm1, double q0, double qp1) const;


	private :
		Solver* solver_;
		std::vector<std::string> M_;

		double recon_time_;
		double reconstruction_time_;
		double transfrom_time_;
		double TBV_time_;
		double selection_time_;

		bool is_log_selection_;
	
		//const int N_stencil_ = 5;
		int N_cand_func_;
		std::vector<std::vector<cellBoundary>> q_bdry_tmp_;
		std::vector<std::vector<double>> TBV_;
		std::vector<cellBoundary> TBV_mem_;
		std::vector<double> TBV_min_;
		int TBV_type_;
		double beta_;
		double T1_;

		//double lambda_Zplus;
};


nW5BVD::nW5BVD(Solver* slv, std::vector<std::string> M, int TBV_type, bool is_log_selection) : solver_(slv), M_(M), TBV_type_(TBV_type), is_log_selection_(is_log_selection) {
	solver_->setReconstruction(this);
	recon_time_ = 0.0;
	reconstruction_time_ = 0.0;
	transfrom_time_ = 0.0;
	TBV_time_ = 0.0;
	selection_time_ = 0.0;


	beta_ = 1.6;
	T1_ = tanh(0.5*beta_);
	/*
	std::cout << "Minor func. used in nW5BVD(sorted): [ ";
	for (auto i : M_) std::cout << i << " ";
	std::cout << "]" << std::endl;
	*/

	int N_max = solver_->getNmax();
	N_cand_func_ = M_.size();
	//printf("N=%d",N_cand_func_);
	q_bdry_tmp_ = std::vector<std::vector<cellBoundary>>(N_cand_func_, std::vector<cellBoundary>(N_max, {0.0, 0.0}));
	TBV_ = std::vector<std::vector<double>>(N_cand_func_, std::vector<double>(N_max, 0.0));
	TBV_mem_ = std::vector<cellBoundary>(N_max, {99, 99});
	TBV_min_ = std::vector<double>(N_max, 1.e+308);

	//lambda_Zplus = pow(solver_->getDX(), 2.0/3.0);
}

nW5BVD::~nW5BVD(void) {
	// std::vector members are automatically deallocated
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void nW5BVD::update(std::vector<cellBoundary>& q_bdry, const std::vector<double>& q) {
	const int N_max = solver_->getNmax();
	const int gs = solver_->getGhostcell();

	double start = omp_get_wtime();
	double time_stamp1 = omp_get_wtime();

	for (int m = 0; m < N_cand_func_; m++) {
		for (int i = gs - 2; i < N_max - gs + 1; i++) {
			if (M_[m] == "THINC") {
				q_bdry_tmp_[m][i+1].L = calcTHINC(q[i-1], q[i], q[i+1]);
				q_bdry_tmp_[m][i+1].R = calcTHINC(q[i+2], q[i+1], q[i]);
			} else {
				q_bdry_tmp_[m][i+1].L = calcWENO5(M_[m], q[i-2], q[i-1], q[i], q[i+1], q[i+2]);
				q_bdry_tmp_[m][i+1].R = calcWENO5(M_[m], q[i+3], q[i+2], q[i+1], q[i], q[i-1]);
			}
		}
	}

	double time_stamp2 = omp_get_wtime();
	if (TBV_type_ == 0) { // simple TBV
		for (int i = gs - 1; i < N_max - gs + 1; i++) {
			double min_val = 1.e+308;
			for (int m = 0; m < N_cand_func_; m++) {
				TBV_[m][i] = fabs(q_bdry_tmp_[m][i].L - q_bdry_tmp_[m][i].R) + fabs(q_bdry_tmp_[m][i+1].L - q_bdry_tmp_[m][i+1].R);
				if (min_val > TBV_[m][i]) {
					min_val = TBV_[m][i];
					TBV_mem_[i].R = m;
					TBV_mem_[i+1].L = m;
				}
			}
			TBV_min_[i] = min_val;
		}
	} else if (TBV_type_ == 1) { // strict TBV1
		for (int i = gs - 1; i < N_max - gs + 1; i++) {
			double min_val = 1.e+308;
			for (int m1 = 0; m1 < N_cand_func_; m1++) {
				for (int m2 = 0; m2 < N_cand_func_; m2++) {
					for (int m3 = 0; m3 < N_cand_func_; m3++) {
						double tmp = fabs(q_bdry_tmp_[m1][i].L - q_bdry_tmp_[m2][i].R) + fabs(q_bdry_tmp_[m2][i+1].L - q_bdry_tmp_[m3][i+1].R);
						if (min_val > tmp) {
							min_val = tmp;
							//TBV_mem_[gs-2] = m1;
							TBV_mem_[i].R = m2;
							TBV_mem_[i+1].L = m2;
							//TBV_mem_[gs] = m3;
						}
					}
				}
			}
			TBV_min_[i] = min_val;
		}
	} else if (TBV_type_ == 2) { // strict TBV2
		for (int i = gs - 1; i < N_max - gs + 1; i++) {
			double min_val = 1.e+308;
			for (int m1 = 0; m1 < N_cand_func_; m1++) {
				for (int m2 = 0; m2 < N_cand_func_; m2++) {
					double tmp = fabs(q_bdry_tmp_[m1][i].L - q_bdry_tmp_[m2][i].R);
					if (min_val > tmp) {
						min_val = tmp;
						//TBV_mem_[gs-2] = m1;
						TBV_mem_[i].L = m1;
						TBV_mem_[i].R = m2;
						//TBV_mem_[gs] = m3;
					}
				}
			}
			TBV_min_[i] = min_val;
		}
	} else if (TBV_type_ == 3) { // strict TBV3
		for (int i = gs - 1; i < N_max - gs + 1; i++) {
			double min_val = 1.e+308;
			for (int m1 = 0; m1 < N_cand_func_; m1++) {
				for (int m2 = 0; m2 < N_cand_func_; m2++) {
					for (int m3 = 0; m3 < N_cand_func_; m3++) {
						for (int m4 = 0; m4 < N_cand_func_; m4++) {
							double tmp = fabs(q_bdry_tmp_[m1][i].L - q_bdry_tmp_[m2][i].R - q_bdry_tmp_[m3][i+1].L + q_bdry_tmp_[m4][i+1].R);
							if (min_val > tmp) {
								min_val = tmp;
								//TBV_mem_[gs-2] = m1;
								TBV_mem_[i].R = m2;
								TBV_mem_[i+1].L = m3;
								//TBV_mem_[gs] = m3;
							}
						}
					}
				}
			}
			TBV_min_[i] = min_val;
		}
	}
	

	double time_stamp3 = omp_get_wtime();

	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		int idx_min_l, idx_min_r;
		if (TBV_type_ == 2) {
			idx_min_l = TBV_mem_[i].L;
			idx_min_r = TBV_mem_[i].R;
			q_bdry[i].L = q_bdry_tmp_[idx_min_l][i].L;
			q_bdry[i].R = q_bdry_tmp_[idx_min_r][i].R;
		} else {
			idx_min_l = TBV_mem_[i+1].L;
			idx_min_r = TBV_mem_[i].R;
			q_bdry[i+1].L = q_bdry_tmp_[idx_min_l][i+1].L;
			q_bdry[i].R = q_bdry_tmp_[idx_min_r][i].R;
		}
	}

	if (is_log_selection_ && TBV_type_ != 2) {
		for (int i = gs - 1; i < N_max - gs + 1; i++) {
			for (int m = 0; m < N_cand_func_; m++) {
				double diff = fabs(TBV_[m][i] - TBV_min_[i]);
				double eps = 0.05;
				if (diff < 1.0e-8) {
					solver_->setSelector(i, m, 1.0-eps);
				} else {
					solver_->setSelector(i, m, eps);
				}
			}
		}
	}

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


double nW5BVD::calcWENO5(std::string m, double qm2, double qm1, double q0, double qp1, double qp2) const {
	double qL; // L value is to i+1/2
	double ul0, ul1, ul2, w0, w1, w2, alp0, alp1, alp2, beta0, beta1, beta2;
	double gamma0, gamma1, u00, u01, u02, u10, u11, u12; // for WENO-ZA
	double eps = 1.0e-20, tmp0, tmp1;
	double ideal_weight0 = 0.1, ideal_weight1 = 0.6, ideal_weight2 = 0.3;
	ul0 = (2.0*qm2 - 7.0*qm1 + 11.0*q0)/6.0;
	ul1 = (-qm1 + 5.0*q0 + 2.0*qp1)/6.0;
	ul2 = (2.0*q0 + 5.0*qp1 - qp2)/6.0;

	//printf("m = %s",m);
	//std::cout << "m = " << m << std::endl;

	

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
		u00 = (qm2 - 4.0*qm1 + 3.0*q0)*0.5; // 1階微分の2次精度後退差分
		beta0 = gamma0*u00*u00 + gamma1*u10*u10;

		u11 = qm1 - 2.0*q0 + qp1; // 2階微分の2次精度中心差分
		u01 = -(qm1 - qp1)*0.5; // 1階微分の2次精度中心差分
		beta1 = gamma0*u01*u01 + gamma1*u11*u11;

		u12 = q0 - 2.0*qp1 + qp2; // 
		u02 = -(3.0*q0 - 4.0*qp1 + qp2)*0.5; // 1階微分の2次精度前進差分
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
		//printf("JS\n");
		alp0 = ideal_weight0/((beta0+eps)*(beta0+eps));
		alp1 = ideal_weight1/((beta1+eps)*(beta1+eps));
		alp2 = ideal_weight2/((beta2+eps)*(beta2+eps));
	} else if (m == "Z" || m == "eta") {
		//printf("Z or eta\n");
		double gsi = fabs(beta0-beta2);
		tmp0 = gsi/(beta0+eps);
		alp0 = ideal_weight0*(1.0 + tmp0*tmp0);
		tmp0 = gsi/(beta1+eps);
		alp1 = ideal_weight1*(1.0 + tmp0*tmp0);
		tmp0 = gsi/(beta2+eps);
		alp2 = ideal_weight2*(1.0 + tmp0*tmp0);
	} else if (m == "Zplus") {
		//printf("Zplus\n");
		double gsi = fabs(beta0-beta2);
		double lambda_Zplus = pow(solver_->getDX(), 2.0/3.0);
		tmp0 = (gsi+eps)/(beta0+eps);
		alp0 = ideal_weight0*(1.0 + tmp0*tmp0 + lambda_Zplus/tmp0);
		tmp0 = (gsi+eps)/(beta1+eps);
		alp1 = ideal_weight1*(1.0 + tmp0*tmp0 + lambda_Zplus/tmp0);
		tmp0 = (gsi+eps)/(beta2+eps);
		alp2 = ideal_weight2*(1.0 + tmp0*tmp0 + lambda_Zplus/tmp0);
	} else if (m == "ZA") {
		//printf("ZA\n");
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

double nW5BVD::calcTHINC(double qm1, double q0, double qp1) const {
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