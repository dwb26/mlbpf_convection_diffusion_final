#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_interp.h>
#include "solvers.h"
#include "particle_filters.h"

// This is the convection model

const double CFL_CONST = 0.8;

int weighted_double_cmp(const void * a, const void * b) {

	struct weighted_double d1 = * (struct weighted_double *) a;
	struct weighted_double d2 = * (struct weighted_double *) b;

	if (d1.x < d2.x)
		return -1;
	if (d2.x < d1.x)
		return 1;
	return 0;
}


double * construct_space_mesh(size_t size, double space_left, double dx, int nx) {
	double * xs = (double *) malloc(size);
	xs[0] = space_left - dx;
	for (int j = 1; j < nx + 2; j++)
		xs[j] = space_left + (j - 1) * dx;
	return xs;
}


void regression_fit(double * s, double * corrections, int N0, int N1, int M_poly, double * poly_weights, double * PHI, double * C, double * C_inv, double * MP, gsl_matrix * C_gsl, gsl_permutation * p, gsl_matrix * C_inv_gsl) {

	/* Set the values of the design matrix */
	int counter = 0, sg;
	for (int n = 0; n < N1; n++) {
		for (int m = 0; m < M_poly; m++)
			PHI[n * M_poly + m] = pow(s[N0 + counter], m);
		counter++;
	}

	/* Do C = PHI.T * PHI */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < M_poly; k++) {
			C[j * M_poly + k] = 0.0;
			for (int n = 0; n < N1; n++)
				C[j * M_poly + k] += PHI[n * M_poly + j] * PHI[n * M_poly + k];
		}
	}

	/* Invert C */
	for (int m = 0; m < M_poly * M_poly; m++)
		C_gsl->data[m] = C[m];
	gsl_linalg_LU_decomp(C_gsl, p, &sg);
	gsl_linalg_LU_invert(C_gsl, p, C_inv_gsl);
	counter = 0;
	for (int m = 0; m < M_poly; m++) {
		for (int n = 0; n < M_poly; n++) {
			C_inv[counter] = gsl_matrix_get(C_inv_gsl, m, n);
			counter++;
		}
	}

	/* Do C_inv * PHI.T */
	for (int j = 0; j < M_poly; j++) {
		for (int k = 0; k < N1; k++) {
			MP[j * N1 + k] = 0.0;
			for (int n = 0; n < M_poly; n++)
				MP[j * N1 + k] += C_inv[j * M_poly + n] * PHI[k * M_poly + n];
		}
	}

	/* Compute the polynomial weights */
	for (int j = 0; j < M_poly; j++) {
		poly_weights[j] = 0.0;
		for (int n = 0; n < N1; n++)
			poly_weights[j] += MP[j * N1 + n] * corrections[n];
	}

}


double poly_eval(double x, double * poly_weights, int poly_degree) {
	double y_hat = 0.0;
	for (int m = 0; m < poly_degree + 1; m++)
		y_hat += poly_weights[m] * pow(x, m);
	return y_hat;
}


void resample(long size, double * w, long * ind, gsl_rng * r) {

	/* Generate the exponentials */
	double * e = (double *) malloc((size + 1) * sizeof(double));
	double g = 0;
	for (long i = 0; i <= size; i++) {
		e[i] = gsl_ran_exponential(r, 1.0);
		g += e[i];
	}
	/* Generate the uniform order statistics */
	double * u = (double *) malloc((size + 1) * sizeof(double));
	u[0] = 0;
	for (long i = 1; i <= size; i++)
		u[i] = u[i - 1] + e[i - 1] / g;

	/* Do the actual sampling with C_inv_gsl cdf */
	double cdf = w[0];
	long j = 0;
	for (long i = 0; i < size; i++) {
		while (cdf < u[i + 1]) {
			j++;
			cdf += w[j];
		}
		ind[i] = j;
	}

	free(e);
	free(u);
}


void random_permuter(int *permutation, int N, gsl_rng *r) {
  
  for (int i = 0; i < N; i++)
    permutation[i] = i;
  
  int j;
  int tmp;
  for (int i = N - 1; i > 0; i--) {
    j = (int)gsl_rng_uniform_int(r, i + 1);
    tmp = permutation[j];
    permutation[j] = permutation[i];
    permutation[i] = tmp;
  }
  
}


double sigmoid(double x, double a, double b) {
	a -= b;
	return a / (1.0 + exp(-0.1 * M_PI * x)) + b;
}


double sigmoid_inv(double x, double a, double b) {
	a -= b;
	return log((x - b) / (a + b - x)) / (0.1 * M_PI);
}


void mutate(int N_tot, double * s, double * s_res, double sig_sd, gsl_rng * rng, int n) {
	for (int i = 0; i < N_tot; i++)
		s[i] = 0.9999 * s_res[i] + gsl_ran_gaussian(rng, sig_sd);
}


void generate_adaptive_artificial_mesh(int N_tot, double * s, int mesh_size, double * s_mesh) {
	double s_lo = 10000.0, s_hi = -10000.0;
	double mesh_incr;
	for (int i = 0; i < N_tot; i++) {
		s_lo = s_lo < s[i] ? s_lo : s[i];
		s_hi = s_hi > s[i] ? s_hi : s[i];
	}
	mesh_incr = (s_hi - s_lo) / (double) (mesh_size - 1);
	for (int l = 0; l < mesh_size; l++)
		s_mesh[l] = s_lo + l * mesh_incr;
}


void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios) {
	

	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx0 = nxs[0], nx1 = nxs[1];
	int nt = hmm->nt;
	int obs_pos0 = nx0;
	int obs_pos1 = nx1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	int poly_degree = 0, M_poly = poly_degree + 1, mesh_size = 1000;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double upper_bound = hmm->upper_bound, lower_bound = hmm->lower_bound;
	double space_left = hmm->space_left, space_right = hmm->space_right;
	double space_length = space_right - space_left;
	double T_stop = hmm->T_stop;
	double dx1 = space_length / (double) (nx1 - 1);
	double dx0 = space_length / (double) (nx0 - 1);	
	double dt = T_stop / (double) (nt - 1);
	double r1 = 0.5 * dt / (dx1 * dx1);
	double r0 = 0.5 * dt / (dx0 * dx0);
	double rdx_sq1 = r1 * dx1 * dx1;
	double rdx_sq0 = r0 * dx0 * dx0;	
	double v = hmm->v;
	double obs, normaliser, abs_normaliser, x_hat, g0, g1, sign_rat;
	double a1 = -r1 * (v * dx1 + 1);
	double b1 = 1 + 2 * r1 + r1 * v * dx1;
	double c1 = r1 * (v * dx1 + 1);
	double d1 = 1 - 2 * r1 - r1 * v * dx1;
	double a0 = -r0 * (v * dx0 + 1);
	double b0 = 1 + 2 * r0 + r0 * v * dx0;
	double c0 = r0 * (v * dx0 + 1);
	double d0 = 1 - 2 * r0 - r0 * v * dx0;
	short * signs = (short *) malloc(N_tot * sizeof(short));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	int * permutation = (int *) malloc(N_tot * sizeof(int));
	double * s = (double *) malloc(N_tot * sizeof(double));
	double * s_sig = (double *) malloc(N_tot * sizeof(double));
	double * s_res = (double *) malloc(N_tot * sizeof(double));
	double * weights = (double *) malloc(N_tot * sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * solns1 = (double *) malloc(N1 * sizeof(double));
	double * solns0 = (double *) malloc(N_tot * sizeof(double));
	double * g1s = (double *) malloc(N1 * sizeof(double));
	double * g0s = (double *) malloc(N1 * sizeof(double));	
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * poly_weights = (double *) malloc(M_poly * sizeof(double));
	double * x_hats = (double *) malloc(length * sizeof(double));
	double * s_mesh = (double *) malloc(mesh_size * sizeof(double));	
	
	
	/* Solver arrays */
	/* ------------- */
	gsl_interp * rho_interp = gsl_interp_alloc(gsl_interp_linear, nx1 + 2);
	gsl_interp_accel * acc = gsl_interp_accel_alloc();
	double * xs1 = construct_space_mesh((nx1 + 2) * sizeof(double), space_left, dx1, nx1);
	double * xs0 = construct_space_mesh((nx0 + 2) * sizeof(double), space_left, dx0, nx0);
	double * rho_c = (double *) malloc((nx1 + 2) * sizeof(double));


	/* Regression matrices */
	/* ------------------- */
	double * PHI = (double *) malloc(N1 * M_poly * sizeof(double));
	double * C = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * C_inv = (double *) malloc(M_poly * M_poly * sizeof(double));
	double * MP = (double *) malloc(N1 * M_poly * sizeof(double));
	gsl_matrix * C_gsl = gsl_matrix_alloc(M_poly, M_poly);
	gsl_permutation * p = gsl_permutation_alloc(M_poly);
	gsl_matrix * C_inv_gsl = gsl_matrix_alloc(M_poly, M_poly);


	/* Level 1 Crank-Nicolson matrices */
	/* ------------------------------- */
	gsl_vector * lower1 = gsl_vector_calloc(nx1 + 1);
	gsl_vector * main1 = gsl_vector_calloc(nx1 + 2);
	gsl_vector * upper1 = gsl_vector_calloc(nx1 + 1);
	gsl_matrix * B1 = gsl_matrix_calloc(nx1 + 2, nx1 + 2);
	gsl_vector * rho1 = gsl_vector_calloc(nx1 + 2);
	gsl_vector * rho_tilde1 = gsl_vector_calloc(nx1 + 2);
	gsl_vector * rho_init1 = gsl_vector_calloc(nx1 + 2);
	construct_present_mat(B1, nx1, c1, d1, r1);
	construct_forward_mat(main1, upper1, lower1, nx1, a1, b1, r1);


	/* Level 0 Crank-Nicolson matrices */
	/* ------------------------------- */
	gsl_vector * lower0 = gsl_vector_calloc(nx0 + 1);
	gsl_vector * main0 = gsl_vector_calloc(nx0 + 2);
	gsl_vector * upper0 = gsl_vector_calloc(nx0 + 1);
	gsl_matrix * B0 = gsl_matrix_calloc(nx0 + 2, nx0 + 2);
	gsl_vector * rho0 = gsl_vector_calloc(nx0 + 2);
	gsl_vector * rho_tilde0 = gsl_vector_calloc(nx0 + 2);
	gsl_vector * rho_init0 = gsl_vector_calloc(nx0 + 2);
	construct_present_mat(B0, nx0, c0, d0, r0);
	construct_forward_mat(main0, upper0, lower0, nx0, a0, b0, r0);


	/* Initial conditions */
	/* ------------------ */
	double s_init = sigmoid_inv(hmm->signal[0], upper_bound, lower_bound);
	for (int i = 0; i < N_tot; i++) {
		s[i] = s_init + gsl_ran_gaussian(rng, sig_sd);
		res_signs[i] = 1;	
	}


	/* Files */
	/* ----- */
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	// FILE * CORRECTIONS = fopen("corrections.txt", "w");
	// FILE * REGRESSION_CURVE = fopen("regression_curve.txt", "w");
	// FILE * TRUE_CURVE = fopen("true_curve.txt", "w");
	// FILE * TRUE_CURVE0 = fopen("true_curve0.txt", "w");
	// fprintf(REGRESSION_CURVE, "%d\n", mesh_size);
	// FILE * LEVEL1_FINE = fopen("level1_fine.txt", "w");
	// FILE * LEVEL1_COARSE = fopen("level1_coarse.txt", "w");
	// FILE * LEVEL0_COARSE = fopen("level0_coarse.txt", "w");
	// FILE * ML_DISTR = fopen("ml_distr.txt", "w");
	// fprintf(ML_DISTR, "%d %d %d\n", N0, N1, N_tot);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0, abs_normaliser = 0.0;
		for (int i = 0; i < N_tot; i++)
			s_sig[i] = sigmoid(s[i], upper_bound, lower_bound);




		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 1 solutions																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			/* Reset the initial conditions to the current time level for the next particle weighting */
			gsl_vector_memcpy(rho1, rho_init1);
			gsl_vector_memcpy(rho0, rho_init0);

			/* Fine solution */
			solve(nx1, nt, dx1, dt, B1, rho1, rho_tilde1, s_sig[i], rdx_sq1, main1, upper1, lower1, CURVE_DATA);
			solns1[i - N0] = rho1->data[obs_pos1];

			/* Coarse solution */
			solve(nx0, nt, dx0, dt, B0, rho0, rho_tilde0, s_sig[i], rdx_sq0, main0, upper0, lower0, CURVE_DATA);
			solns0[i] = rho0->data[obs_pos0];
			
			/* Record the corrections samples for the regression approximation to the true correction curve */
			corrections[i - N0] = solns1[i - N0] - solns0[i];
			// fprintf(CORRECTIONS, "%e %e\n", s_sig[i], corrections[i - N0]);

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 0 solutions 																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N0; i++) {

			gsl_vector_memcpy(rho0, rho_init0);
			solve(nx0, nt, dx0, dt, B0, rho0, rho_tilde0, s_sig[i], rdx_sq0, main0, upper0, lower0, CURVE_DATA);
			solns0[i] = rho0->data[obs_pos0];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Regression corrections																					 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		if (N1 > 0) {
			regression_fit(s_sig, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);
			for (int i = 0; i < N_tot; i++)
				solns0[i] += poly_eval(s_sig[i], poly_weights, poly_degree);

			// generate_adaptive_artificial_mesh(N_tot, s_sig, mesh_size, s_mesh);
			// for (int l = 0; l < mesh_size; l++) {

				// gsl_vector_memcpy(rho1, rho_init1);
				// gsl_vector_memcpy(rho0, rho_init0);

				/* Output the regressed correction curve approximation over the artificial particle mesh */
				// fprintf(REGRESSION_CURVE, "%.16e %.16e\n", s_mesh[l], poly_eval(s_mesh[l], poly_weights, poly_degree));

				/* Output the true correction curve */
				// solve(nx1, nt, dx1, dt, B1, rho1, rho_tilde1, s_mesh[l], rdx_sq1, main1, upper1, lower1, CURVE_DATA);
			// 	g1 = rho1->data[obs_pos1];
			// 	solve(nx0, nt, dx0, dt, B0, rho0, rho_tilde0, s_mesh[l], rdx_sq0, main0, upper0, lower0, CURVE_DATA);
			// 	g0 = rho0->data[obs_pos0];
			// 	g0 += poly_eval(s_mesh[l], poly_weights, poly_degree);
			// 	// fprintf(TRUE_CURVE, "%.16e ", g1 - g0);
			// 	fprintf(TRUE_CURVE, "%.16e ", g1);
			// 	fprintf(TRUE_CURVE0, "%.16e ", g0);

			// }
			// fprintf(TRUE_CURVE, "\n");
			// fprintf(TRUE_CURVE0, "\n");

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight assignment																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */

		/* Level 1 */
		/* ------- */
		for (int i = N0; i < N_tot; i++) {

			g1s[i - N0] = gsl_ran_gaussian_pdf(solns1[i - N0] - obs, obs_sd);
			g0s[i - N0] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd);
			weights[i] = (g1s[i - N0] - g0s[i - N0]) / (double) N1;

			// fprintf(LEVEL1_FINE, "%e %e\n", solns1[i - N0], g1s[i - N0]);
			// fprintf(LEVEL1_COARSE, "%e %e\n", solns0[i], g0s[i - N0]);

			// fprintf(LEVEL1_FINE, "%e %e\n", s[i], g1s[i - N0]);
			// fprintf(LEVEL1_COARSE, "%e %e\n", s[i], g0s[i - N0]);

		}

		/* Level 0 */
		/* ------- */
		for (int i = 0; i < N0; i++) {
			weights[i] = gsl_ran_gaussian_pdf(solns0[i] - obs, obs_sd) / (double) N0;

			// fprintf(LEVEL0_COARSE, "%e %e\n", solns0[i], weights[i] * N0);
			// fprintf(LEVEL0_COARSE, "%e %e\n", s[i], weights[i] * N0);

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*	 																										 																										 */
		/* Normalisation 																						 	 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N_tot; i++) {

			/* Scale the weights by the previous sign and compute the new sign */
			weights[i] *= res_signs[i];
			signs[i] = weights[i] < 0 ? -1 : 1;
			absolute_weights[i] = fabs(weights[i]);

			/* Compute the normalisation terms */
			normaliser += weights[i];
			abs_normaliser += absolute_weights[i];

		}
		// for (int i = N0; i < N_tot; i++) {
			// fprintf(LEVEL1_FINE, "%e %e\n", s[i], g1s[i - N0] / (double) N1 / (double) normaliser);
			// fprintf(LEVEL1_COARSE, "%e %e\n", s[i], g0s[i - N0] / (double) N1 / (double) normaliser);
		// }
		// for (int i = 0; i < N0; i++)
			// fprintf(LEVEL0_COARSE, "%e %e\n", s[i], weights[i] / (double) normaliser);
		x_hat = 0.0;
		for (int i = 0; i < N_tot; i++) {
			weights[i] /= normaliser;
			absolute_weights[i] /= abs_normaliser;
			ml_weighted[n][i].x = s_sig[i];
			ml_weighted[n][i].w = weights[i];
			x_hat += s_sig[i] * weights[i];
		}
		x_hats[n] = x_hat;
		// for (int i = 0; i < N_tot; i++)
		// 	fprintf(ML_DISTR, "%e %e\n", ml_weighted[n][i].x, ml_weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N_tot, absolute_weights, ind, rng);
		random_permuter(permutation, N_tot, rng);
		for (int i = 0; i < N_tot; i++) {
			s_res[permutation[i]] = s[ind[i]];
			res_signs[permutation[i]] = signs[ind[i]];
		}
		mutate(N_tot, s, s_res, sig_sd, rng, n);


		/* Initial condition evolution */
		/* --------------------------- */
		solve(nx1, nt, dx1, dt, B1, rho_init1, rho_tilde1, x_hats[n], rdx_sq1, main1, upper1, lower1, CURVE_DATA);
		for (int j = 0; j < nx1 + 2; j++)
			rho_c[j] = rho_init1->data[j];
		gsl_interp_init(rho_interp, xs1, rho_c, nx1 + 2);
		for (int j = 1; j < nx0 + 1; j++)
			rho_init0->data[j] = gsl_interp_eval(rho_interp, xs1, rho_c, xs0[j], acc);
		rho_init0->data[0] = rho_init1->data[1];
		rho_init0->data[nx0 + 1] = rho_init1->data[nx1 + 1];

	}

	fclose(CURVE_DATA);
	// fclose(CORRECTIONS);
	// fclose(REGRESSION_CURVE);
	// fclose(LEVEL1_FINE);
	// fclose(LEVEL1_COARSE);
	// fclose(LEVEL0_COARSE);
	// fclose(ML_DISTR);
	// fclose(TRUE_CURVE);
	// fclose(TRUE_CURVE0);

	free(signs);
	free(res_signs);
	free(ind);
	free(permutation);
	free(s);
	free(s_sig);
	free(s_res);
	free(weights);
	free(absolute_weights);
	free(solns0);
	free(solns1);
	free(g0s);
	free(g1s);
	free(corrections);
	free(poly_weights);
	free(x_hats);
	free(s_mesh);
	gsl_interp_free(rho_interp);
	gsl_interp_accel_free(acc);
	free(xs0);
	free(xs1);
	free(rho_c);
	free(PHI);
	free(C);
	free(C_inv);
	free(MP);
	gsl_matrix_free(C_gsl);
	gsl_permutation_free(p);
	gsl_matrix_free(C_inv_gsl);
	gsl_vector_free(lower0);
	gsl_vector_free(main0);
	gsl_vector_free(upper0);
	gsl_matrix_free(B0);
	gsl_vector_free(rho0);
	gsl_vector_free(rho_tilde0);
	gsl_vector_free(rho_init0);
	gsl_vector_free(lower1);
	gsl_vector_free(main1);
	gsl_vector_free(upper1);
	gsl_matrix_free(B1);
	gsl_vector_free(rho1);
	gsl_vector_free(rho_tilde1);
	gsl_vector_free(rho_init1);

}


void bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted) {

	
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
	double upper_bound = hmm->upper_bound, lower_bound = hmm->lower_bound;
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
	size_t size = nx * sizeof(double);
	long * ind = (long *) malloc(N * sizeof(long));
	double * s = (double *) malloc(N * sizeof(double));
	double * s_sig = (double *) malloc(N * sizeof(double));
	double * s_res = (double *) malloc(N * sizeof(double));
	double * weights = (double *) malloc(N * sizeof(double));
	double * x_hats = (double *) malloc(length * sizeof(double));


	/* Crank-Nicolson matrices */
	/* ----------------------- */
	/* Construct the forward time matrix */
	gsl_vector * lower = gsl_vector_alloc(nx + 1);
	gsl_vector * main = gsl_vector_alloc(nx + 2);
	gsl_vector * upper = gsl_vector_alloc(nx + 1);
	gsl_matrix * B = gsl_matrix_calloc(nx + 2, nx + 2);
	gsl_vector * rho = gsl_vector_calloc(nx + 2);
	gsl_vector * rho_tilde = gsl_vector_calloc(nx + 2);
	gsl_vector * rho_init = gsl_vector_calloc(nx + 2);
	construct_present_mat(B, nx, c, d, r);
	construct_forward_mat(main, upper, lower, nx, a, b, r);


	/* Initial conditions */
	/* ------------------ */
	for (int i = 0; i < N; i++)
		s[i] = sigmoid_inv(hmm->signal[0], upper_bound, lower_bound) + gsl_ran_gaussian(rng, sig_sd);

	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * BPF_XHATS = fopen("bpf_xhats.txt", "w");
	// FILE * BPF_DISTR = fopen("bpf_distr.txt", "w");
	// FILE * RES_BPF_DISTR = fopen("res_bpf_distr.txt", "w");	
	// fprintf(BPF_DISTR, "%d %d\n", length, N);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0;



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight generation																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N; i++) {

			s_sig[i] = sigmoid(s[i], upper_bound, lower_bound);
			gsl_vector_memcpy(rho, rho_init);
			solve(nx, nt, dx, dt, B, rho, rho_tilde, s_sig[i], rdx_sq, main, upper, lower, CURVE_DATA);
			weights[i] = gsl_ran_gaussian_pdf(rho->data[obs_pos] - obs, obs_sd);
			normaliser += weights[i];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Normalisation 																							 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = s_sig[i];
			weighted[n][i].w = weights[i];
			x_hat += s_sig[i] * weights[i];
		}
		x_hats[n] = x_hat;
		fprintf(BPF_XHATS, "%.16e ", x_hat);
		// for (int i = 0; i < N; i++)
			// fprintf(BPF_DISTR, "%.16e %.16e\n", weighted[n][i].x, weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			s_res[i] = s[ind[i]];
		// for (int i = 0; i < N; i++)
			// fprintf(RES_BPF_DISTR, "%.16e ", sigmoid(s_res[i], upper_bound, lower_bound));
		mutate(N, s, s_res, sig_sd, rng, n);

		/* Initial condition update */
		solve(nx, nt, dx, dt, B, rho_init, rho_tilde, x_hats[n], rdx_sq, main, upper, lower, CURVE_DATA);

	}

	fclose(CURVE_DATA);
	fclose(BPF_XHATS);
	// fclose(BPF_DISTR);
	// fclose(RES_BPF_DISTR);

	free(ind);
	free(s);
	free(s_sig);
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


void bootstrap_particle_filter_var_nx(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted, int nx) {

	
	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nt = hmm->nt;
	int obs_pos = nx;
	double sig_sd = hmm->sig_sd;
	double obs_sd = hmm->obs_sd;
	double upper_bound = hmm->upper_bound, lower_bound = hmm->lower_bound;
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
	size_t size = nx * sizeof(double);
	long * ind = (long *) malloc(N * sizeof(long));
	double * s = (double *) malloc(N * sizeof(double));
	double * s_sig = (double *) malloc(N * sizeof(double));
	double * s_res = (double *) malloc(N * sizeof(double));
	double * weights = (double *) malloc(N * sizeof(double));
	double * x_hats = (double *) malloc(length * sizeof(double));


	/* Crank-Nicolson matrices */
	/* ----------------------- */
	/* Construct the forward time matrix */
	gsl_vector * lower = gsl_vector_alloc(nx + 1);
	gsl_vector * main = gsl_vector_alloc(nx + 2);
	gsl_vector * upper = gsl_vector_alloc(nx + 1);
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
		s[i] = sigmoid_inv(s0, upper_bound, lower_bound) + gsl_ran_gaussian(rng, sig_sd);

	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * BPF_DISTR = fopen("bpf_distr.txt", "w");
	fprintf(BPF_DISTR, "%d\n", N);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0;



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight generation																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N; i++) {

			s_sig[i] = sigmoid(s[i], upper_bound, lower_bound);
			gsl_vector_memcpy(rho, rho_init);
			solve(nx, nt, dx, dt, B, rho, rho_tilde, s_sig[i], rdx_sq, main, upper, lower, CURVE_DATA);
			weights[i] = gsl_ran_gaussian_pdf(rho->data[obs_pos] - obs, obs_sd);
			normaliser += weights[i];

		}




		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Normalisation 																							 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = s_sig[i];
			weighted[n][i].w = weights[i];
			x_hat += s_sig[i] * weights[i];
		}
		x_hats[n] = x_hat;
		// printf("n = %d, x_hat = %lf, sig = %lf\n", n, x_hat, hmm->signal[n]);
		// for (int i = 0; i < N; i++)
		// 	fprintf(BPF_DISTR, "%e %e\n", weighted[n][i].x, weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			s_res[i] = s[ind[i]];
		mutate(N, s, s_res, sig_sd, rng, n);

		/* Initial condition update */
		solve(nx, nt, dx, dt, B, rho_init, rho_tilde, x_hats[n], rdx_sq, main, upper, lower, CURVE_DATA);

	}

	fclose(CURVE_DATA);
	fclose(BPF_DISTR);

	free(ind);
	free(s);
	free(s_sig);
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
	double upper_bound = hmm->upper_bound, lower_bound = hmm->lower_bound;
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
	size_t size = nx * sizeof(double);
	long * ind = (long *) malloc(N * sizeof(long));
	double * s = (double *) malloc(N * sizeof(double));
	double * s_sig = (double *) malloc(N * sizeof(double));
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
		s[i] = sigmoid_inv(s0, upper_bound, lower_bound) + gsl_ran_gaussian(rng, sig_sd);


	/* Files */
	/* ----- */
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * REF_PARTICLES = fopen("ref_particles.txt", "w");
	FILE * NORMALISERS = fopen("normalisers.txt", "w");
	FILE * REF_XHATS = fopen("ref_xhats.txt", "w");
	FILE * RES_REF_DISTR = fopen("res_ref_distr.txt", "w");
	fprintf(REF_PARTICLES, "%d %d\n", length, N);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0;


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Weight generation																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N; i++) {

			s_sig[i] = sigmoid(s[i], upper_bound, lower_bound);
			gsl_vector_memcpy(rho, rho_init);
			solve(nx, nt, dx, dt, B, rho, rho_tilde, s_sig[i], rdx_sq, main, upper, lower, CURVE_DATA);
			weights[i] = gsl_ran_gaussian_pdf(rho->data[obs_pos] - obs, obs_sd);
			normaliser += weights[i];			

		}
		fprintf(NORMALISERS, "%e ", normaliser);


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Normalisation 																							 																							 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = s_sig[i];
			weighted[n][i].w = weights[i];
			x_hat += s_sig[i] * weights[i];
		}
		x_hats[n] = x_hat;
		fprintf(REF_XHATS, "%e ", x_hat);
		printf("ref xhat(%d) = %lf\n", n, x_hat);
		// for (int i = 0; i < N; i++)
			// fprintf(REF_PARTICLES, "%.16e %.16e\n", weighted[n][i].x, weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			s_res[i] = s[ind[i]];
		// for (int i = 0; i < N; i++)
			// fprintf(RES_REF_DISTR, "%.16e ", sigmoid(s_res[i], upper_bound, lower_bound));
			// fprintf(REF_PARTICLES, "%.16e ", s_res[i]);
		// fprintf(REF_PARTICLES, "\n");
		mutate(N, s, s_res, sig_sd, rng, n);

		/* Initial condition update */
		solve(nx, nt, dx, dt, B, rho_init, rho_tilde, x_hats[n], rdx_sq, main, upper, lower, CURVE_DATA);

	}

	fclose(CURVE_DATA);
	fclose(REF_PARTICLES);
	fclose(NORMALISERS);
	fclose(REF_XHATS);
	fclose(RES_REF_DISTR);

	free(ind);
	free(s);
	free(s_sig);
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




