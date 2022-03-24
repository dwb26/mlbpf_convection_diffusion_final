void generate_ics(double * ics, double dx, int nx, double space_left);

void construct_forward(double r, int nx, gsl_vector * lower, gsl_vector * main, gsl_vector * upper);

void construct_present(double r, int nx, gsl_matrix * B);

void minmod_convection_diffusion_solver(double * rho, int nx, double dx, double dt, int obs_pos, double T_stop, FILE * CURVE_DATA, double v, double s, double r, gsl_vector * lower, gsl_vector * main, gsl_vector * upper, gsl_matrix * B, gsl_vector * u, gsl_vector * Bu, double * slopes, double * Q_star);

void construct_present_mat(gsl_matrix * B, int nx, double c, double d, double r);

void construct_forward_mat(gsl_vector * main, gsl_vector * upper, gsl_vector * lower, int nx, double a, double b, double r);

void solve(int nx, int nt, double dx, double dt, gsl_matrix * B, gsl_vector * rho, gsl_vector * rho_tilde, double s, double rdx_sq, gsl_vector * main, gsl_vector * upper, gsl_vector * lower, FILE * CURVE_DATA);

void level0_solve(int nx, int nt, double dx, double dt, gsl_matrix * B, gsl_vector * rho, gsl_vector * rho_tilde, double s, double rdx_sq, gsl_vector * main, gsl_vector * upper, gsl_vector * lower, FILE * CURVE_DATA0);