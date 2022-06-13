	typedef struct {
	double * signal;
	double * observations;
	int length;
	int nx;
	int nt;
	double sig_sd;
	double obs_sd;
	double space_left;
	double space_right;
	double T_stop;
	double v;
	double mu;
	double upper_bound;
	double lower_bound;
	int N_LEVELS;
	int N_MESHES;
	int N_ALLOCS;
} HMM;

typedef struct weighted_double {
	double x;
	double w;
} w_double;

int weighted_double_cmp(const void * a, const void * b);

void bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted);

void bootstrap_particle_filter_var_nx(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted, int nx);

void ref_bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted);

void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios);

void ml_bootstrap_particle_filter_timed(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios);

double sigmoid(double x, double a, double b);

double sigmoid_inv(double x, double a, double b);