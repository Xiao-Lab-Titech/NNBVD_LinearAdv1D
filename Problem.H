#ifndef _PROBLEM_H_
#define _PROBLEM_H_

#include "Solver.H"
#include "BoundaryCondition.H"
#include "Exact.H"

class Problem {
    public :
        Problem(void) {}
        Problem(Solver* slv) : solver_(slv) {solver_->setProblem(this);}
        ~Problem(void) {}

        virtual void initProblem(double* q, double* qe) = 0;
        virtual void setIdx(int idx) {idx_ = idx;}
        virtual void setTE(double te) {te_ = te;}

        virtual std::string getName() const = 0;
        virtual int getIdx() const = 0;
        virtual bool hasExact() const = 0;
        virtual double getXL() const = 0;
        virtual double getXR() const = 0;
        virtual double getTE() const = 0;
        

    private :
        Solver* solver_;
        double xl_, xr_;
        double te_;

        int idx_;
        std::string name_;
        bool has_exact_;
        BoundaryCondition* bc_;
        Exact* exact_;
};

class LinearAdv1DProb : public Problem {
    using InitFunc = void (LinearAdv1DProb::*)(double* q, double* qe);

    public :
        LinearAdv1DProb(Solver* slv) : solver_(slv),
            init_table_{&LinearAdv1DProb::sineWave,
                        &LinearAdv1DProb::squareWave,
                        &LinearAdv1DProb::JiangAndShu,
                        &LinearAdv1DProb::randomSine,
                        &LinearAdv1DProb::randomSquare,
                        &LinearAdv1DProb::randomTriangle,
                        &LinearAdv1DProb::randomPoly,
                        &LinearAdv1DProb::nonlinearDiscontinuity,
                        &LinearAdv1DProb::cubedSine } {}
        ~LinearAdv1DProb(void) {}

        void initProblem(double* q, double* qe) override;
        void setIdx(int idx) {idx_ = idx;}
        void setTE(double te) {te_ = te;}

        std::string getName() const {return name_;}
        int getIdx() const {return idx_;}
        bool hasExact() const {return has_exact_;}
        double getXL() const {return xl_;}
        double getXR() const {return xr_;}
        double getTE() const {return te_;}

    private :
        void sineWave(double* q, double* qe);
        void squareWave(double* q, double* qe);
        void JiangAndShu(double* q, double* qe);
        void randomSine(double* q, double* qe);
        void randomSquare(double* q, double* qe);
        void randomTriangle(double* q, double* qe);
        void randomPoly(double* q, double* qe);
        void nonlinearDiscontinuity(double* q, double* qe);
        void cubedSine(double* q, double* qe);

    private :
        Solver* solver_;
        double xl_, xr_;
        double te_;
        std::vector<InitFunc> init_table_; 

        int idx_;
        std::string name_;
        bool has_exact_;
        BoundaryCondition* bc_;
        Exact* exact_;
};


#endif