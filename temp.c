void ref_bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx = hmm->nx;
	int nt = hmm->nt;
	int obs_pos = nx;
	double sig_sd = hmm->sig_sd;
	double obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right;
	double T_stop = hmm->T_stop;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double dt = T_stop / (double) (nt - 1);
	double r = 0.5 * dt / (dx * dx);
	double rdx_sq = r * dx * dx;
	double v = hmm->v;	
	double obs, normaliser, x_hat;
	double a = -r * (v * dx + 1);
	double b = 1 + 2 * r + r * v * dx;
	double c = r * (v * dx + 1);
	double d = 1 - 2 * r - r * v * dx;
	long * ind = (long *) malloc(N * sizeof(long));
	double * s = (double *) malloc(N * sizeof(double));
	double * s_res = (double *) malloc(N * sizeof(double));
	double * weights = (double *) malloc(N * sizeof(double));
	double * x_hats = (double *) malloc(length * sizeof(double));


	/* Crank-Nicolson matrices */
	/* ----------------------- */
	gsl_vector * lower = gsl_vector_calloc(nx + 1);
	gsl_vector * main = gsl_vector_calloc(nx + 2);
	gsl_vector * upper = gsl_vector_calloc(nx + 1);
	gsl_matrix * B = gsl_matrix_calloc(nx + 2, nx + 2);
	gsl_vector * rho = gsl_vector_calloc(nx + 2);
	gsl_vector * rho_tilde = gsl_vector_calloc(nx + 2);
	gsl_vector * rho_init = gsl_vector_calloc(nx + 2);
	construct_present_mat(B, nx, c, d, r);
	construct_forward_mat(main, upper, lower, nx, a, b, r);


	/* Initial conditions */
	/* ------------------ */
	double s0 = hmm->signal[0];
	for (int i = 0; i < N; i++)
		s[i] = s0 + gsl_ran_gaussian(rng, sig_sd);


	/* Files */
	/* ----- */
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * BPF_PARTICLES = fopen("bpf_particles.txt", "w");
	FILE * NORMALISERS = fopen("normalisers.txt", "w");
	FILE * X_HATS = fopen("x_hats.txt", "w");
	FILE * BPF_DISTR = fopen("bpf_distr.txt", "w");
	fprintf(BPF_PARTICLES, "%d %d\n", length, N);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0;


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Weight generation																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N; i++) {

			solve(nx, nt, dx, dt, B, rho, rho_tilde, s[i], rdx_sq, main, upper, lower, CURVE_DATA);
			weights[i] = gsl_ran_gaussian_pdf(rho->data[obs_pos] - obs, obs_sd);
			normaliser += weights[i];
			gsl_vector_memcpy(rho, rho_init);

		}
		fprintf(NORMALISERS, "%e ", normaliser);


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Normalisation 																							 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = s[i];
			weighted[n][i].w = weights[i];
			x_hat += s[i] * weights[i];
		}
		x_hats[n] = x_hat;
		fprintf(X_HATS, "%e ", x_hat);
		printf("ref xhat(%d) = %lf\n", n, x_hat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			s_res[i] = s[ind[i]];
		for (int i = 0; i < N; i++)
			fprintf(BPF_PARTICLES, "%e ", s_res[i]);
		fprintf(BPF_PARTICLES, "\n");
		mutate(N, s, s_res, sig_sd, rng, n);

		/* Initial condition update */
		solve(nx, nt, dx, dt, B, rho, rho_tilde, x_hats[n], rdx_sq, main, upper, lower, CURVE_DATA);
		gsl_vector_memcpy(rho_init, rho);

	}

	fclose(CURVE_DATA);
	fclose(BPF_PARTICLES);
	fclose(NORMALISERS);
	fclose(X_HATS);
	fclose(BPF_DISTR);	

	free(ind);
	free(s);
	free(s_res);
	free(weights);
	free(x_hats);

	gsl_vector_free(lower);
	gsl_vector_free(main);
	gsl_vector_free(upper);
	gsl_matrix_free(B);
	gsl_vector_free(rho);
	gsl_vector_free(rho_tilde);
	gsl_vector_free(rho_init);

}