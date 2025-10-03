#ifndef _MLBASEDW3TBVD_H_
#define _MLBASEDW3TBVD_H_ 
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <omp.h>
#include "../Reconstruction.H"
#include "../Solver.H"
#include "../Parameter.H"

class MLBasedW3TBVD : public Reconstruction 
{
	public :
		MLBasedW3TBVD(Solver* slv, const char* onnx_path, bool isGPU);
		~MLBasedW3TBVD(void);
		void update(cellBoundary** q_bdry, const double* const* q);
		std::string getName() const {std::string s = "MLBasedW3TBVD";return s;}
		double getReconstructionTime() const {return recon_time_;}
		double getTransformTime() const {return transform_time_;}
		double getInferenceTime() const {return inference_time_;}
		double getSelectionTime() const {return selection_time_;}

	private :
		void inference(const double* const* q);
		double calcTHINC(double qm1, double q0, double qp1) const;
		double calcWENO3(double qm1, double q0, double qp1) const;

	private :
		Solver* solver_;

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
		const int N_stencil_ = 3;
		double* stencil_;
		float** y_;
		double** q_tmp_;
		
};


MLBasedW3TBVD::MLBasedW3TBVD(Solver* slv, const char* onnx_path, bool isGPU) : solver_(slv) {
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
    const int N_variable = 1;
    const int N_input = 5;
    const int N_output = 1;
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
	initVector1d(stencil_, 5, 0.0);
	initVector2d(y_, 3, N_max, 0.0f);
	initVector2d(q_tmp_, 3, N_max, 0.0);

}

MLBasedW3TBVD::~MLBasedW3TBVD(void) {
    //int N_max = solver_->getNmax();
	freeVector1d(input_shapes_);
	freeVector1d(output_shapes_);
	freeVector1d(input_data_);
	freeVector1d(output_data_);

	freeVector1d(stencil_);
	freeVector2d(y_, 3);
	freeVector2d(q_tmp_, 3);
}

void MLBasedW3TBVD::update(cellBoundary** q_bdry, const double* const* q) {
	const int N_max = solver_->getNmax();
	const int gs = solver_->getGhostcell();
	const double gamma = solver_->getGamma();
	double** stencil;
	initVector2d<double>(stencil, 3, N_stencil_ + 1, 0.0);

	//std::chrono::system_clock::time_point  start, end;
	//start = std::chrono::system_clock::now(); // 計測開始
	double start = omp_get_wtime();

	Reconstruction::transform(RECONSTRUCTION_VAR, solver_, q_tmp_, q);
	
	double time_stamp1 = omp_get_wtime();
	inference(q_tmp_);
	double time_stamp2 = omp_get_wtime();
	
	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		for (int j = 0; j < N_stencil_ + 1; j++) {
			stencil[0][j] = q_tmp_[0][i - (N_stencil_-1)/2 + j];
			stencil[1][j] = q_tmp_[1][i - (N_stencil_-1)/2 + j];
			stencil[2][j] = q_tmp_[2][i - (N_stencil_-1)/2 + j];
		}

		if (RECONSTRUCTION_VAR == 0 && IS_CHARACTERISTIC_DECOMP) Reconstruction::cons2charX(stencil, N_stencil_, q_tmp_, gamma, i, i+1);
		if (RECONSTRUCTION_VAR == 1 && IS_CHARACTERISTIC_DECOMP) Reconstruction::prim2charX(stencil, N_stencil_, q_tmp_, gamma, i, i+1);
		
		for (int j = 0; j < 3; j++) {
			if (y_[j][i] < KAPPA_REF) {
				q_bdry[j][i+1].L = calcWENO3(stencil[j][0], stencil[j][1], stencil[j][2]);
			} else {
				q_bdry[j][i+1].L = calcTHINC(stencil[j][0], stencil[j][1], stencil[j][2]);
			}
			if (y_[j][i+1] < KAPPA_REF) {
				q_bdry[j][i+1].R = calcWENO3(stencil[j][3], stencil[j][2], stencil[j][1]);
			} else {
				q_bdry[j][i+1].R = calcTHINC(stencil[j][3], stencil[j][2], stencil[j][1]);
			}
		}
	}


	for (int i = gs - 1; i < N_max - gs + 1; i++) {
		if (RECONSTRUCTION_VAR == 0 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2consX(q_bdry, q_tmp_, gamma, i-1, i, i);
		if (RECONSTRUCTION_VAR == 1 && IS_CHARACTERISTIC_DECOMP) Reconstruction::char2primX(q_bdry, q_tmp_, gamma, i-1, i, i);
	}

	freeVector2d<double>(stencil, 3);
	/*
	for (int i = 0; i < 3; i++) {
		for (int j = gs - 1; j < N_max - gs + 1; j++) {
			if (y_[i][j] < KAPPA_REF) {
				q_bdry[i][j+1].L = calcWENO3(q_tmp_[i][j-1], q_tmp_[i][j], q_tmp_[i][j+1]);
				q_bdry[i][j].R = calcWENO3(q_tmp_[i][j+1], q_tmp_[i][j], q_tmp_[i][j-1]);
			} else {
				q_bdry[i][j+1].L = calcTHINC(q_tmp_[i][j-1], q_tmp_[i][j], q_tmp_[i][j+1]);
				q_bdry[i][j].R = calcTHINC(q_tmp_[i][j+1], q_tmp_[i][j], q_tmp_[i][j-1]);
			}
		}
	}
	*/
	double end = omp_get_wtime(); 
	//end = std::chrono::system_clock::now(); // 計測終了
	double elapsed = end - start;


	//double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
	//printf("recon time: %f\n", elapsed);
	
	transform_time_ += time_stamp1 - start;
	inference_time_ += time_stamp2 - time_stamp1;
	selection_time_ += end - time_stamp2;
	recon_time_ += elapsed;
}

inline void MLBasedW3TBVD::inference(const double* const* q) {
	const int N_max = solver_->getNmax();
	const int gs = solver_->getGhostcell();
	int N_input = 0;

	for (int i = 0; i < 3; i++) {
		for (int j = gs; j < N_max - gs; j++) {
			/*
			for (int k = 0; k < 5; k++) {
				stencil_[k] = q[i][j + k - 2];
			}
			*/
			
			stencil_[0] = q[i][j - 2];
			stencil_[1] = q[i][j - 1];
			stencil_[2] = q[i][j];
			stencil_[3] = q[i][j + 1];
			stencil_[4] = q[i][j + 2];

			double M = -1.0e308, m = 1.0e308;
			for (int k = 0; k < 5; k++) {
				if (stencil_[k] > M) M = stencil_[k];
				if (stencil_[k] < m) m = stencil_[k];
			}
			float diff = M - m;
			//float monotone_indicator = ((stencil_[2]-stencil_[1])*(stencil_[3]-stencil_[2]) < 0) ? 
			//							0.0 : 1.0;

			if (diff > 1.0e-15) {
				float inv_diff = 1.f/diff;
				for (int k = 0; k < 5; k++) {
					input_data_[N_input] = (float)(stencil_[k] - m)*inv_diff;
					N_input++;
				}
				//input_data_[N_input] = monotone_indicator;
				//N_input++;
			} else {
				for (int k = 0; k < 5; k++) {
					input_data_[N_input] = 0.0;
					N_input++;
				}
				//input_data_[N_input] = monotone_indicator;
				//N_input++;
			}
			//printf("j = %d, [%f ,%f ,%f ,%f ,%f ,%f]\n", j, input_data_[N_input-6], input_data_[N_input-5],input_data_[N_input-4],input_data_[N_input-3],input_data_[N_input-2],input_data_[N_input-1]);
		}
	}

	session_.Run(run_options_, &input_names_ , &input_tensor_, 1U, &output_names_ , &output_tensor_, 1U);

	N_input = 0;
	for (int i = 0; i < 3; i++) {
		for (int j = gs; j < N_max - gs; j++) {
			y_[i][j] = output_data_[N_input++];
		}
	}
}


double MLBasedW3TBVD::calcTHINC(double qm1, double q0, double qp1) const {
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

double MLBasedW3TBVD::calcWENO3(double qm1, double q0, double qp1) const {
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