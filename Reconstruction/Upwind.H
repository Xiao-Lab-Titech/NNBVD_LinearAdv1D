#ifndef _Upwind_H_
#define _Upwind_H_ 

#include "../Reconstruction.H"
#include "../Solver.H"
#include "../RawVector.H"

class Upwind : public Reconstruction 
{
	public :
		Upwind(Solver* slv);
		~Upwind(void);
		void update(cellBoundary* q_bdry, const double* q);
		std::string getName() const {std::string s = "Upwind";return s;}
		double getReconstructionTime() const {return recon_time_;}

	private :
		Solver* solver_;
		
		double recon_time_;
		
};


Upwind::Upwind(Solver* slv) : solver_(slv) {
	solver_->setReconstruction(this);
}

Upwind::~Upwind(void) {

}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
void Upwind::update(cellBoundary* q_bdry, const double* q) {
	int N_max = solver_->getNmax();
	int gs = solver_->getGhostcell();

	for (int i = gs - 1; i < N_max - gs; i++) {
		q_bdry[i+1].L = q[i];
		q_bdry[i+1].R = q[i+1];
	}
}



#endif 