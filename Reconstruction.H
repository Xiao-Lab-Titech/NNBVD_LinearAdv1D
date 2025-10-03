#ifndef _RECONSTRUCTION_H_
#define _RECONSTRUCTION_H_
#include "Parameter.H"
#include "RawVector.H"

class Solver;

class Reconstruction 
{
	public :
		Reconstruction(void) {};
		Reconstruction(Solver* slv) : solver_(slv) {solver_->setReconstruction(this);}
		~Reconstruction(void) {};
		virtual void update(cellBoundary* q_bdry, const double* q) = 0;
		virtual std::string getName() const = 0;
		virtual double getReconstructionTime() const = 0;
		virtual std::vector<std::string> getCandidateFunctionLabels() const {return std::vector<std::string> {};}


		protected :



	private :
		Solver* solver_;

};


#endif
