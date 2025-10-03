#ifndef _MLBASEDNW5BVD_H_
#define _MLBASEDNW5BVD_H_ 
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"

class MLBasednW5BVD : public Reconstruction 
{
	public :
		MLBasednW5BVD(Solver* slv, const char* onnx_path, bool isGPU, std::vector<std::string> M);
		~MLBasednW5BVD(void);
		void update(cellBoundary* q_bdry, const double* q);
		std::string getName() const {std::string s = "MLBasednWBVD";return s;}
		double getReconstructionTime() const {return recon_time_;}
		double getTransformTime() const {return transform_time_;}
		double getInferenceTime() const {return inference_time_;}
		double getSelectionTime() const {return selection_time_;}

	private :
		void inference(const double* q);
		double calcWENO5(std::string m, double qm2, double qm1, double q0, double qp1, double qp2) const;
		double calcTHINC(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;
		std::vector<std::string> M_;

		double recon_time_;
		double transform_time_;
		double inference_time_;
		double selection_time_;


		Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "test" };
		Ort::SessionOptions session_options_;
		Ort::Session session_{ nullptr };
		Ort::RunOptions run_options_;

		float* input_data_;
		float* output_data_;
		int64_t* input_shapes_;
		int64_t* output_shapes_;

		Ort::Value input_tensor_{nullptr};
		Ort::Value output_tensor_{nullptr};

		const char* input_names_ = "input";
		const char* output_names_ = "output";

		double beta_;
		double T1_;
		int N_input_ = 16;
		//int N_output_ = 2;
		const int N_stencil_ = 5;
        int N_cand_func_;
		double* stencil_;
		float* y_;
		double* q_tmp_;
		
};


MLBasednW5BVD::MLBasednW5BVD(Solver* slv, const char* onnx_path, bool isGPU, std::vector<std::string> M) : solver_(slv), M_(M) {
	solver_->setReconstruction(this);
	recon_time_ = 0.0;
	transform_time_ = 0.0;
	inference_time_ = 0.0;
	selection_time_ = 0.0;

    // apriori setup
	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	//session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    std::vector<std::string> avaliableProviders = Ort::GetAvailableProviders();
    auto cudaAvaliable = std::find(avaliableProviders.begin(), avaliableProviders.end(), "CUDAExecutionProvider");
    for (auto i : avaliableProviders) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    if (cudaAvaliable != avaliableProviders.end() && isGPU) {
        std::cout << "CUDA Inferrence. " << std::endl;
        OrtCUDAProviderOptions cudaOptions;
        session_options_.AppendExecutionProvider_CUDA(cudaOptions);
		//cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::HEURISTIC; //OrtCudnnConvAlgoSearch();

    } else {
        std::cout << "CPU Inferrence." << std::endl;
		int N_threads = 4; // 2 or 4
		//session_options_.SetInterOpNumThreads(N_threads);
		session_options_.SetIntraOpNumThreads(N_threads);
    }

    session_ = Ort::Session(env_, onnx_path, session_options_);

    // make data
    int N_max = solver_->getNmax();
    N_cand_func_ = M_.size();
    const int N_variable = 1;
    const int N_input = N_input_;
    const int N_output = N_cand_func_;
	int N_input_shapes = 2;
	int N_output_shapes = 2;
	int N_input_data = N_max*N_variable*N_input;
	int N_output_data = N_max*N_variable*N_output;
	initVector1d(input_shapes_, N_input_shapes, (int64_t)0);
	initVector1d(output_shapes_, N_output_shapes, (int64_t)0);
    input_shapes_[0] = N_variable*N_max; input_shapes_[1] = N_input; 
    output_shapes_[0] = N_variable*N_max; output_shapes_[1] = N_output;

	initVector1d(input_data_, N_input_data, 0.f);
	initVector1d(output_data_, N_output_data, 0.f);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_data_, N_input_data, input_shapes_, N_input_shapes);
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, output_data_, N_output_data, output_shapes_, N_output_shapes);

	beta_ = 1.6;
	T1_ = tanh(0.5*beta_);
	initVector1d(stencil_, N_stencil_+2, 0.0);
	initVector1d(y_, N_max, 0.0f);

}

MLBasednW5BVD::~MLBasednW5BVD(void) {
    //int N_max = solver_->getNmax();
	freeVector1d(input_shapes_);
	freeVector1d(output_shapes_);
	freeVector1d(input_data_);
	freeVector1d(output_data_);

	freeVector1d(stencil_);
	freeVector1d(y_);
}

void MLBasednW5BVD::update(cellBoundary* q_bdry, const double* q) {
	const int N_max = solver_->getNmax();
	const int gs = solver_->getGhostcell();

	double start = omp_get_wtime();

	
	double time_stamp1 = omp_get_wtime();
	inference(q);
	double time_stamp2 = omp_get_wtime();
	
	for (int i = gs - 1; i < N_max - gs + 1; i++) {
        std::string pred_label = M_[y_[i]];
		//std::cout << i << " " << pred_label << std::endl;
        if (pred_label == "THINC") {
			q_bdry[i+1].L = calcTHINC(q[i-1], q[i], q[i+1]);
			q_bdry[i].R = calcTHINC(q[i+1], q[i], q[i-1]);
        } else {
            q_bdry[i+1].L = calcWENO5(pred_label, q[i-2], q[i-1], q[i], q[i+1], q[i+2]);
            q_bdry[i].R = calcWENO5(pred_label, q[i+2], q[i+1], q[i], q[i-1], q[i-2]);  
        }
		//solver_->setSelector(i, y_[i], 1.0);
	}


	double end = omp_get_wtime(); 
	double elapsed = end - start;


	
	transform_time_ += time_stamp1 - start;
	inference_time_ += time_stamp2 - time_stamp1;
	selection_time_ += end - time_stamp2;
	recon_time_ += elapsed;
}

inline void MLBasednW5BVD::inference(const double* q) {
	const int N_max = solver_->getNmax();
	const int gs = solver_->getGhostcell();
	int N_input = 0;

	double *diff_f1, *diff_f2, *diff_b2, *diff_c2, *diff_c2_2;
	initVector1d<double>(diff_f1, N_max, 0.0);
	initVector1d<double>(diff_f2, N_max, 0.0);
	initVector1d<double>(diff_b2, N_max, 0.0);
	initVector1d<double>(diff_c2, N_max, 0.0);
	initVector1d<double>(diff_c2_2, N_max, 0.0);

	for (int i = 2; i < N_max-2; i++) {
		diff_f1[i] = fabs(q[i+1]-q[i]);
		diff_c2[i] = fabs(q[i+1]-q[i-1])*0.5;
		diff_f2[i] = fabs(3.0*q[i]-4.0*q[i+1]+q[i+2])*0.5;
		diff_b2[i] = fabs(q[i-2]-4.0*q[i-1]+3.0*q[i])*0.5;
		diff_c2_2[i] = fabs(q[i-1]-2.0*q[i]+q[i+1]);
	}
	double M_f1, M_f2, M_c2, M_b2, M_c2_2;
	M_f1 = *std::max_element(diff_f1 + 2, diff_f1 + N_max - 2);
	M_f2 = *std::max_element(diff_f2 + 2, diff_f2 + N_max - 2);
	M_c2 = *std::max_element(diff_c2 + 2, diff_c2 + N_max - 2);
	M_b2 = *std::max_element(diff_b2 + 2, diff_b2 + N_max - 2);
	M_c2_2 = *std::max_element(diff_c2_2 + 2, diff_c2_2 + N_max - 2);


    double* in;
    initVector1d(in, N_input_, 0.0);

    for (int i = gs; i < N_max - gs; i++) {
        for (int j = 0; j < N_stencil_ + 2; j++) {
            stencil_[j] = q[i + j - (N_stencil_+2-1)/2];
        }
        
        in[0] = fabs(stencil_[4]-stencil_[3])/M_f1;
        in[1] = fabs(stencil_[3]-stencil_[2])/M_f1;
        in[2] = fabs(stencil_[3]-stencil_[1])/M_c2*0.5;
        in[3] = fabs(stencil_[4]-stencil_[2])/M_c2*0.5;
        in[4] = fabs(stencil_[5]-stencil_[3])/M_c2*0.5;
        in[5] = fabs(-(3.0*stencil_[2]-4.0*stencil_[3]+stencil_[4]))/M_f2*0.5;
        in[6] = fabs(-(3.0*stencil_[3]-4.0*stencil_[4]+stencil_[5]))/M_f2*0.5;
        in[7] = fabs(-(3.0*stencil_[4]-4.0*stencil_[5]+stencil_[6]))/M_f2*0.5;
        in[8] = fabs(stencil_[0]-4.0*stencil_[1]+3.0*stencil_[2])/M_b2*0.5;
        in[9] = fabs(stencil_[1]-4.0*stencil_[2]+3.0*stencil_[3])/M_b2*0.5;
        in[10] = fabs(stencil_[2]-4.0*stencil_[3]+3.0*stencil_[4])/M_b2*0.5;
        in[11] = fabs(stencil_[0]-2.0*stencil_[1]+stencil_[2])/M_c2_2;
        in[12] = fabs(stencil_[1]-2.0*stencil_[2]+stencil_[3])/M_c2_2;
        in[13] = fabs(stencil_[2]-2.0*stencil_[3]+stencil_[4])/M_c2_2;
        in[14] = fabs(stencil_[3]-2.0*stencil_[4]+stencil_[5])/M_c2_2;
        in[15] = fabs(stencil_[4]-2.0*stencil_[5]+stencil_[6])/M_c2_2;
        
        //double diff_max = std::max(fabs(in[0]), std::max(fabs(in[1]), 1.e-15));
        //in[16] = std::log10(diff_max);
		for (int j = 0; j < N_input_; j++) {
			input_data_[N_input] = (float)in[j];
			N_input++;
		}
		/*
        for (int j = 0; j < N_input_; j++) {
            in[j] = fabs(in[j]);
            //in[j] = (in[j])/diff_max;
        }

        if (diff_max > 1.0e-15) {
            for (int j = 0; j < N_input_; j++) {
                input_data_[N_input] = (float)in[j];
                N_input++;
            }
        } else {
            for (int j = 0; j < N_input_; j++) {
                if (j!=16) in[j] = 0.0;
                input_data_[N_input] = (float)in[j];
                N_input++;
            }
        }
		*/
        //printf("j = %d, [%f ,%f ,%f ,%f ,%f ,%f]\n", j, input_data_[N_input-6], input_data_[N_input-5],input_data_[N_input-4],input_data_[N_input-3],input_data_[N_input-2],input_data_[N_input-1]);
    }

    freeVector1d(in);

	session_.Run(run_options_, &input_names_ , &input_tensor_, 1U, &output_names_ , &output_tensor_, 1U);

	N_input = 0;
    double pred, tmp;
    int idx_max;
    for (int i = gs; i < N_max - gs; i++) {
        pred = 0.0;
        idx_max = 0;
        for (int j = 0; j < N_cand_func_; j++) {
            tmp = output_data_[N_input++];
            if (pred < tmp) {
                pred = tmp;
                idx_max = j;
            }
        }
        y_[i] = idx_max;
    }
}

double MLBasednW5BVD::calcWENO5(std::string m, double qm2, double qm1, double q0, double qp1, double qp2) const {
	double qL; // L value is to i+1/2
	double ul0, ul1, ul2, w0, w1, w2, alp0, alp1, alp2, beta0, beta1, beta2;
	double gamma0, gamma1, u00, u01, u02, u10, u11, u12; // for WENO-ZA
	double eps = 1.0e-15, tmp0, tmp1;
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

double MLBasednW5BVD::calcTHINC(double qm1, double q0, double qp1) const {
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