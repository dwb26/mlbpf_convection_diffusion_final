#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
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
	return a / (1.0 + exp(-0.01 * M_PI * x)) + b;
}


double sigmoid_inv(double x, double a, double b) {
	return log((x - b) / (a + b - x)) / (0.01 * M_PI);
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
	int obs_pos0 = nx0 + 1;
	int obs_pos1 = nx1 + 1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	int poly_degree = 2, M_poly = poly_degree + 1, mesh_size = 1000;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double upper_bound = hmm->upper_bound, lower_bound = hmm->lower_bound;
	double space_left = hmm->space_left, space_right = hmm->space_right;
	double space_length = space_right - space_left;
	double T_stop = hmm->T_stop;
	double dx0 = space_length / (double) (nx0 - 1);
	double dx1 = space_length / (double) (nx1 - 1);
	double dt = T_stop / (double) (nt - 1);
	double r0 = 0.5 * dt / (dx0 * dx0);
	double r1 = 0.5 * dt / (dx1 * dx1);
	double rdx_sq0 = r0 * dx0 * dx0;
	double rdx_sq1 = r1 * dx1 * dx1;
	double v = hmm->v;
	double obs, normaliser, abs_normaliser, x_hat, g0, g1, sign_rat;
	double a0 = -r0 * (v * dx0 + 1);
	double b0 = 1 + 2 * r0 + r0 * v * dx0;
	double c0 = r0 * (v * dx0 + 1);
	double d0 = 1 - 2 * r0 - r0 * v * dx0;
	double a1 = -r1 * (v * dx1 + 1);
	double b1 = 1 + 2 * r1 + r1 * v * dx1;
	double c1 = r1 * (v * dx1 + 1);
	double d1 = 1 - 2 * r1 - r1 * v * dx1;
	short * signs = (short *) malloc(N_tot * sizeof(short));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	int * permutation = (int *) malloc(N_tot * sizeof(int));
	double * s = (double *) malloc(N_tot * sizeof(double));
	double * s_sig = (double *) malloc(N_tot * sizeof(double));
	double * s_res = (double *) malloc(N_tot * sizeof(double));
	double * weights = (double *) malloc(N_tot * sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * solns0 = (double *) malloc(N_tot * sizeof(double));
	double * solns1 = (double *) malloc(N1 * sizeof(double));
	double * g0s = (double *) malloc(N1 * sizeof(double));
	double * g1s = (double *) malloc(N1 * sizeof(double));
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * poly_weights = (double *) malloc(M_poly * sizeof(double));
	double * x_hats = (double *) malloc(length * sizeof(double));
	double * s_mesh = (double *) malloc(mesh_size * sizeof(double));	
	
	
	/* Solver arrays */
	/* ------------- */
	gsl_interp * rho_interp = gsl_interp_alloc(gsl_interp_linear, nx1 + 2);
	gsl_interp_accel * acc = gsl_interp_accel_alloc();
	double * xs0 = construct_space_mesh((nx0 + 2) * sizeof(double), space_left, dx0, nx0);
	double * xs1 = construct_space_mesh((nx1 + 2) * sizeof(double), space_left, dx1, nx1);
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
	generate_adaptive_artificial_mesh(N_tot, s, mesh_size, s_mesh);


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


	/* Initial conditions */
	/* ------------------ */
	double s0 = hmm->signal[0];
	for (int i = 0; i < N_tot; i++) {
		s[i] = sigmoid_inv(s0, upper_bound, lower_bound) + gsl_ran_gaussian(rng, sig_sd);
		res_signs[i] = 1;	
	}


	/* Files */
	/* ----- */
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * ML_XHATS = fopen("ml_xhats.txt", "w");
	FILE * CORRECTIONS = fopen("corrections.txt", "w");
	FILE * REGRESSION_CURVE = fopen("regression_curve.txt", "w");
	FILE * TRUE_CURVE = fopen("true_curve.txt", "w");
	FILE * TRUE_CURVE0 = fopen("true_curve0.txt", "w");
	FILE * LEVEL1_FINE = fopen("level1_fine.txt", "w");
	FILE * LEVEL1_COARSE = fopen("level1_coarse.txt", "w");
	FILE * LEVEL0_COARSE = fopen("level0_coarse.txt", "w");
	FILE * ML_DISTR = fopen("ml_distr.txt", "w");
	FILE * SIGNS = fopen("signs.txt", "w");
	fprintf(REGRESSION_CURVE, "%d\n", mesh_size);
	fprintf(ML_DISTR, "%d %d %d\n", N0, N1, N_tot);
	fprintf(SIGNS, "%d %d %d\n", N0, N1, N_tot);



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		normaliser = 0.0, abs_normaliser = 0.0;



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 1 solutions																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			/* Reset the initial conditions to the current time level for the next particle weighting */
			s_sig[i] = sigmoid(s[i], upper_bound, lower_bound);
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
			fprintf(CORRECTIONS, "%e %e\n", s_sig[i], corrections[i - N0]);

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Level 0 solutions 																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N0; i++) {

			s_sig[i] = sigmoid(s[i], upper_bound, lower_bound);
			gsl_vector_memcpy(rho0, rho_init0);
			solve(nx0, nt, dx0, dt, B0, rho0, rho_tilde0, s[i], rdx_sq0, main0, upper0, lower0, CURVE_DATA);
			solns0[i] = rho0->data[obs_pos0];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Regression corrections																					 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		if (N1 > 0) {
			;
			regression_fit(s, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);
			for (int i = 0; i < N_tot; i++)
				solns0[i] += poly_eval(s_sig[i], poly_weights, poly_degree);

			generate_adaptive_artificial_mesh(N_tot, s_sig, mesh_size, s_mesh);
			for (int l = 0; l < mesh_size; l++) {

				gsl_vector_memcpy(rho1, rho_init1);
				gsl_vector_memcpy(rho0, rho_init0);

				/* Output the regressed correction curve approximation over the artificial particle mesh */
				fprintf(REGRESSION_CURVE, "%e %e\n", sigmoid(s_mesh[l], upper_bound, lower_bound), poly_eval(s_mesh[l], poly_weights, poly_degree));

				/* Output the true correction curve */
				solve(nx1, nt, dx1, dt, B1, rho1, rho_tilde1, sigmoid(s_mesh[l], upper_bound, lower_bound), rdx_sq1, main1, upper1, lower1, CURVE_DATA);
				g1 = rho1->data[obs_pos1];
				solve(nx0, nt, dx0, dt, B0, rho0, rho_tilde0, sigmoid(s_mesh[l], upper_bound, lower_bound), rdx_sq0, main0, upper0, lower0, CURVE_DATA);
				g0 = rho0->data[obs_pos0];
				// g0 = rho0[obs_pos0] + poly_eval(s_mesh[l], poly_weights, poly_degree);
				// fprintf(TRUE_CURVE, "%e ", g1 - g0);
				fprintf(TRUE_CURVE, "%e ", g1);
				fprintf(TRUE_CURVE0, "%e ", g0);

			}
			fprintf(TRUE_CURVE, "\n");
			fprintf(TRUE_CURVE0, "\n");

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
		// 	fprintf(LEVEL1_FINE, "%e %e\n", s[i], g1s[i - N0] / (double) N1 / (double) normaliser);
		// 	fprintf(LEVEL1_COARSE, "%e %e\n", s[i], g0s[i - N0] / (double) N1 / (double) normaliser);
		// }
		// for (int i = 0; i < N0; i++)
		// 	fprintf(LEVEL0_COARSE, "%e %e\n", s[i], weights[i] / (double) normaliser);

		x_hat = 0.0, sign_rat = 0.0;
		for (int i = 0; i < N_tot; i++) {
			weights[i] /= normaliser;
			absolute_weights[i] /= abs_normaliser;
			ml_weighted[n][i].x = s_sig[i];
			ml_weighted[n][i].w = weights[i];
			x_hat += s_sig[i] * weights[i];
		}
		fprintf(ML_XHATS, "%e ", x_hat);
		x_hats[n] = x_hat;
		// sign_ratios[n] = sign_rat;
		// fprintf(SIGNS, "\n");
		// for (int i = 0; i < N_tot; i++)
		// 	fprintf(ML_DISTR, "%e %e\n", ml_weighted[n][i].x, ml_weighted[n][i].w);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 																										 */
		/* Resample and mutate 																						 																					 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N_tot, absolute_weights, ind, rng);
		for (int i = 0; i < N_tot; i++) {
			s_res[i] = s[ind[i]];
			res_signs[i] = signs[ind[i]];
		}
		mutate(N_tot, s, s_res, sig_sd, rng, n);
		solve(nx1, nt, dx1, dt, B1, rho_init1, rho_tilde1, x_hats[n], rdx_sq1, main1, upper1, lower1, CURVE_DATA);
		gsl_interp_init(rho_interp, xs1, rho_c, nx1 + 2);
		for (int j = 1; j < nx0 + 1; j++)
			rho_init0->data[j] = gsl_interp_eval(rho_interp, xs1, rho_c, xs0[j], acc);
		rho_init0->data[0] = rho_init1->data[1];
		rho_init0->data[nx0 + 1] = rho_init1->data[nx1 + 1];

	}

	fclose(CURVE_DATA);
	fclose(ML_XHATS);
	fclose(CORRECTIONS);
	fclose(REGRESSION_CURVE);
	fclose(TRUE_CURVE);
	fclose(TRUE_CURVE0);
	fclose(LEVEL1_FINE);
	fclose(LEVEL1_COARSE);
	fclose(LEVEL0_COARSE);
	fclose(ML_DISTR);
	fclose(SIGNS);

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
	double s0 = hmm->signal[0];
	for (int i = 0; i < N; i++)
		s[i] = sigmoid_inv(s0, upper_bound, lower_bound) + gsl_ran_gaussian(rng, sig_sd);

	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * X_HATS = fopen("x_hats.txt", "w");
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
			solve(nx, nt, dx, dt, B, rho, rho_tilde, s_sig[i], rdx_sq, main, upper, lower, CURVE_DATA);
			weights[i] = gsl_ran_gaussian_pdf(rho->data[obs_pos] - obs, obs_sd);
			normaliser += weights[i];
			gsl_vector_memcpy(rho, rho_init);

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
		fprintf(X_HATS, "%e ", x_hat);
		// printf("n = %d, x_hat = %lf\n", n, x_hat);
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
		solve(nx, nt, dx, dt, B, rho, rho_tilde, x_hats[n], rdx_sq, main, upper, lower, CURVE_DATA);
		gsl_vector_memcpy(rho_init, rho);

	}

	fclose(CURVE_DATA);
	fclose(X_HATS);
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
		/*																											 																										 */
		/* Weight generation																						 																						 */
		/*																											 																										 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N; i++) {

			s_sig[i] = sigmoid(s[i], upper_bound, lower_bound);
			solve(nx, nt, dx, dt, B, rho, rho_tilde, s_sig[i], rdx_sq, main, upper, lower, CURVE_DATA);
			weights[i] = gsl_ran_gaussian_pdf(rho->data[obs_pos] - obs, obs_sd);
			normaliser += weights[i];
			gsl_vector_memcpy(rho, rho_init);

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















// void ml_bootstrap_particle_filter_debug(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, FILE * L2_ERR_DATA) {
	

// 	/* --------------------------------------------------- Setup --------------------------------------------------- */
// 	/* ------------------------------------------------------------------------------------------------------------- */

// 	/* General parameters */
// 	/* ------------------ */
// 	int length = hmm->length;
// 	int nx0 = nxs[0], nx1 = nxs[1];
// 	int obs_pos0 = nx0 + 1;
// 	int obs_pos1 = nx1 + 1;
// 	int lag = hmm->lag, start_point = 0, counter0, counter1;
// 	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
// 	int poly_degree = 2, M_poly = poly_degree + 1;
// 	double sign_rat = 0.0, coarse_scaler = 0.5;
// 	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
// 	double v = hmm->v, mu = hmm->mu;
// 	double space_left = hmm->space_left, space_right = hmm->space_right;
// 	double space_length = space_right - space_left;
// 	double T_stop = hmm->T_stop;
// 	double dx0 = space_length / (double) (nx0 - 1);
// 	double dx1 = space_length / (double) (nx1 - 1);
// 	double dt0 = CFL_CONST * dx0 / v;
// 	double dt1 = CFL_CONST * dx1 / v;
// 	double r0 = mu * dt0 / (2.0 * dx0 * dx0), r1 = mu * dt1 / (2.0 * dx1 * dx1);
// 	double obs, normaliser, abs_normaliser, g0, g1, ml_xhat;
// 	size_t size0 = (nx0 + 4) * sizeof(double), size1 = (nx1 + 4) * sizeof(double);
// 	short * signs = (short *) malloc(N_tot * sizeof(short));
// 	short * res_signs = (short *) malloc(N_tot * sizeof(short));
// 	long * ind = (long *) malloc(N_tot * sizeof(long));
// 	double * s = (double *) malloc(N_tot * sizeof(double));
// 	double * s_res = (double *) malloc(N_tot * sizeof(double));
// 	double * weights = (double *) malloc(N_tot * sizeof(double));
// 	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
// 	double * solns1 = (double *) malloc(N1 * sizeof(double));
// 	double * solns0 = (double *) malloc(N1 * sizeof(double));
// 	double * corrections = (double *) malloc(N1 * sizeof(double));
// 	double * poly_weights = (double *) malloc((poly_degree + 1) * sizeof(double));
// 	double * theta = (double *) calloc(2, sizeof(double));
// 	double * xhats = (double *) malloc(length * sizeof(double));
// 	double * rho0 = (double *) calloc(nx0 + 4, sizeof(double));
// 	double * rho1 = (double *) calloc(nx1 + 4, sizeof(double));
// 	double * ics0 = (double *) calloc(nx0 + 4, sizeof(double));
// 	double * ics1 = (double *) calloc(nx1 + 4, sizeof(double));
// 	double * slopes0 =  (double *) malloc((nx0 + 1) * sizeof(double));
// 	double * Q_star0 = (double *) malloc(nx0 * sizeof(double));
// 	double * slopes1 =  (double *) malloc((nx1 + 1) * sizeof(double));
// 	double * Q_star1 = (double *) malloc(nx1 * sizeof(double));
// 	double * xs0 = construct_space_mesh((nx0 + 4) * sizeof(double), space_left, dx0, nx0);
// 	double * xs1 = construct_space_mesh((nx1 + 4) * sizeof(double), space_left, dx1, nx1);
// 	double * PHI = (double *) malloc(N1 * M_poly * sizeof(double));
// 	double * C = (double *) malloc(M_poly * M_poly * sizeof(double));
// 	double * C_inv = (double *) malloc(M_poly * M_poly * sizeof(double));
// 	double * MP = (double *) malloc(N1 * M_poly * sizeof(double));
// 	double ** X = (double **) malloc(N_tot * sizeof(double *));
// 	for (int i = 0; i < N_tot; i++)
// 		X[i] = (double *) malloc((lag + 1) * sizeof(double));


// 	/* Regression matrices */
// 	/* ------------------- */
// 	gsl_matrix * C_gsl = gsl_matrix_alloc(M_poly, M_poly);
// 	gsl_permutation * p = gsl_permutation_alloc(M_poly);
// 	gsl_matrix * C_inv_gsl = gsl_matrix_alloc(M_poly, M_poly);


// 	/* Level 0 Crank-Nicolson matrices */
// 	/* ------------------------------- */
// 	/* Construct the forward time matrix */
// 	gsl_vector * lower0 = gsl_vector_alloc(nx0 + 1);
// 	gsl_vector * main0 = gsl_vector_alloc(nx0 + 2);
// 	gsl_vector * upper0 = gsl_vector_alloc(nx0 + 1);
// 	construct_forward(r0, nx0 + 2, lower0, main0, upper0);

// 	/* Construct the present time matrix */
// 	gsl_matrix * B0 = gsl_matrix_calloc(nx0 + 2, nx0 + 2);
// 	gsl_vector * u0 = gsl_vector_calloc(nx0 + 2);
// 	gsl_vector * Bu0 = gsl_vector_alloc(nx0 + 2);
// 	construct_present(r0, nx0 + 2, B0);


// 	/* Level 1 Crank-Nicolson matrices */
// 	/* ------------------------------- */
// 	/* Construct the forward time matrix */
// 	gsl_vector * lower1 = gsl_vector_alloc(nx1 + 1);
// 	gsl_vector * main1 = gsl_vector_alloc(nx1 + 2);
// 	gsl_vector * upper1 = gsl_vector_alloc(nx1 + 1);
// 	construct_forward(r1, nx1 + 2, lower1, main1, upper1);
	
// 	/* Construct the present time matrix */
// 	gsl_matrix * B1 = gsl_matrix_calloc(nx1 + 2, nx1 + 2);
// 	gsl_vector * u1 = gsl_vector_calloc(nx1 + 2);
// 	gsl_vector * Bu1 = gsl_vector_alloc(nx1 + 2);
// 	construct_present(r1, nx1 + 2, B1);


// 	/* Initial conditions */
// 	/* ------------------ */
// 	double s0 = hmm->signal[0];
// 	generate_ics(ics0, dx0, nx0, space_left);
// 	generate_ics(ics1, dx1, nx1, space_left);
// 	memcpy(rho0, ics0, size0);
// 	memcpy(rho1, ics1, size1);
// 	for (int i = 0; i < N_tot; i++) {
// 		s[i] = gsl_ran_gaussian(rng, sig_sd) + s0;
// 		X[i][0] = s[i];
// 		res_signs[i] = 1;	
// 	}
// 	gsl_interp * ics_interp = gsl_interp_alloc(gsl_interp_linear, nx1 + 4);
// 	gsl_interp_accel * acc = gsl_interp_accel_alloc();

// 	/* Output files */
// 	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
// 	// FILE * XHATS = fopen("xhats.txt", "w");
// 	int Ns = 250, Ns_fine = 2000, i_min, i_max;
// 	double s_min, s_max, s_glob_min, s_glob_max, l2_err = 0.0;
// 	char nx0_str[50], name[100], reg_name[100], cor_name[100];
// 	snprintf(nx0_str, 50, "%d", nx0);
// 	sprintf(name, "corrections_nx0=%s.txt", nx0_str);
// 	sprintf(reg_name, "regression_data_nx0=%s.txt", nx0_str);
// 	sprintf(cor_name, "exact_corrections_nx0=%s.txt", nx0_str);
// 	FILE * CORRECTIONS = fopen(name, "w");
// 	FILE * REGRESSION_DATA = fopen(reg_name, "w");
// 	FILE * EXACT_CORRECTIONS = fopen(cor_name, "w");



// 	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
// 	/* ------------------------------------------------------------------------------------------------------------- */
// 	for (int n = 0; n < length; n++) {

// 		obs = hmm->observations[n];
// 		if (n > lag)
// 			start_point++;

// 		s_glob_min = s[0], s_glob_max = s[N_tot - 1];
// 		for (int i = 0; i < N_tot; i++) {
// 			s_glob_min = s[i] < s_glob_min ? s[i] : s_glob_min;
// 			s_glob_max = s[i] > s_glob_max ? s[i] : s_glob_max;
// 		}


// 		/* --------------------------------------------------------------------------------------------------------- */
// 		/*																											 */
// 		/* Level 1 solutions																						 */
// 		/*																											 */
// 		/* --------------------------------------------------------------------------------------------------------- */
// 		normaliser = 0.0, abs_normaliser = 0.0;
// 		for (int i = N0; i < N_tot; i++) {


// 			/* Fine solution */
// 			/* ------------- */
// 			/* Fine solve with respect to the historical particles */
// 			minmod_convection_diffusion_solver(rho1, nx1, dx1, dt1, obs_pos1, T_stop, CURVE_DATA, v, s[i], r1, lower1, main1, upper1, B1, u1, Bu1, slopes1, Q_star1);
// 			solns1[i - N0] = rho1[obs_pos1];
			

// 			/* Coarse solution */
// 			/* --------------- */
// 			/* Coarse solve with respect to the historical particles */
// 			minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s[i], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
// 			solns0[i - N0] = rho0[obs_pos0];
			
// 			/* Reset the initial conditions to the current time level for the next particle weighting */
// 			corrections[i - N0] = solns1[i - N0] - solns0[i - N0];
// 			memcpy(rho1, ics1, size1);
// 			memcpy(rho0, ics0, size0);

// 			fprintf(CORRECTIONS, "%e %e\n", s[i], corrections[i - N0]);

// 		}

// 		if (N1 > 0) {

// 			/* Determine the regression weights */
// 			regression_fit(s, corrections, N0, N1, M_poly, poly_weights, PHI, C, C_inv, MP, C_gsl, p, C_inv_gsl);

// 			/* Write the regressed correction curve based on the level 1 correction data */
// 			double ds_fine = (s_glob_max - s_glob_min) / (double) (Ns_fine - 1);
// 			for (int i = 0; i < Ns_fine; i++)
// 				fprintf(REGRESSION_DATA, "%e ", s_glob_min + i * ds_fine);
// 			fprintf(REGRESSION_DATA, "\n");
// 			for (int i = 0; i < Ns_fine; i++)
// 				fprintf(REGRESSION_DATA, "%e ", poly_eval(s_glob_min + i * ds_fine, poly_weights, poly_degree));
// 			fprintf(REGRESSION_DATA, "\n");

// 			// THIS IS THE REGRESSION PLOT CODE
// 			/* Find the true correction curve based on the entire data set */
// 			for (int i = 0; i < Ns_fine; i++) {

// 				minmod_convection_diffusion_solver(rho1, nx1, dx1, dt1, obs_pos1, T_stop, CURVE_DATA, v, s_glob_min + i * ds_fine, r1, lower1, main1, upper1, B1, u1, Bu1, slopes1, Q_star1);				
// 				minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s_glob_min + i * ds_fine, r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
				
// 				fprintf(EXACT_CORRECTIONS, "%e %e\n", s_glob_min + i * ds_fine, rho1[obs_pos1] - rho0[obs_pos0]);

// 				/* Reset the initial conditions to the current time level for the next particle weighting */
// 				memcpy(rho1, ics1, size1);
// 				memcpy(rho0, ics0, size0);

// 			}

// 			// THIS IS THE L2 ERROR CODE
// 			/* Find the true correction curve based on the entire data set */
// 			for (int i = 0; i < N_tot; i++) {

// 				minmod_convection_diffusion_solver(rho1, nx1, dx1, dt1, obs_pos1, T_stop, CURVE_DATA, v, s[i], r1, lower1, main1, upper1, B1, u1, Bu1, slopes1, Q_star1);				
// 				minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s[i], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);

// 				l2_err += (rho1[obs_pos1] - rho0[obs_pos0] - poly_eval(s[i], poly_weights, poly_degree)) * (rho1[obs_pos1] - rho0[obs_pos0] - poly_eval(s[i], poly_weights, poly_degree)) / (double) length;

// 				/* Reset the initial conditions to the current time level for the next particle weighting */
// 				memcpy(rho1, ics1, size1);
// 				memcpy(rho0, ics0, size0);

// 			}

// 		}


// 		/* --------------------------------------------------------------------------------------------------------- */
// 		/*																											 */
// 		/* Level 1 weight generation																				 */
// 		/*																											 */
// 		/* --------------------------------------------------------------------------------------------------------- */
// 		for (int i = N0; i < N_tot; i++) {

// 			g1 = gsl_ran_gaussian_pdf(solns1[i - N0] - obs, obs_sd);
// 			g0 = gsl_ran_gaussian_pdf(solns0[i - N0] + poly_eval(s[i], poly_weights, poly_degree) - obs, obs_sd);

// 			weights[i] = (g1 - g0) * (double) res_signs[i] / (double) N1;
// 			absolute_weights[i] = fabs(weights[i]);
// 			normaliser += weights[i];
// 			abs_normaliser += absolute_weights[i];

// 		}



// 		/* --------------------------------------------------------------------------------------------------------- */
// 		/*																											 */
// 		/* Level 0 weight generation																				 */
// 		/*																											 */
// 		/* --------------------------------------------------------------------------------------------------------- */
// 		if (N1 > 0) {

// 			for (int i = 0; i < N0; i++) {


// 				/* Coarse solution */
// 				/* --------------- */
// 				/* Coarse solve with respect to the historical particles */
// 				counter0 = 0;
// 				for (int m = start_point; m < n; m++) {
// 					minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, X[i][counter0], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
// 					counter0++;
// 				}
// 				minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s[i], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
// 				g0 = gsl_ran_gaussian_pdf(rho0[obs_pos0] + poly_eval(s[i], poly_weights, poly_degree) - obs, obs_sd);


// 				/* Weight computation */
// 				/* ------------------ */
// 				weights[i] = g0 * (double) res_signs[i] / (double) N0;
// 				absolute_weights[i] = fabs(weights[i]);
// 				normaliser += weights[i];
// 				abs_normaliser += absolute_weights[i];

// 				/* Reset the initial conditions to the current time level for the next particle weighting */
// 				memcpy(rho0, ics0, size0);

// 			}

// 		}

// 		else {

// 			for (int i = 0; i < N0; i++) {


// 				/* Coarse solution */
// 				/* --------------- */
// 				/* Coarse solve with respect to the historical particles */
// 				counter0 = 0;
// 				for (int m = start_point; m < n; m++) {
// 					minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, X[i][counter0], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
// 					counter0++;
// 				}
// 				minmod_convection_diffusion_solver(rho0, nx0, dx0, dt0, obs_pos0, T_stop, CURVE_DATA, v, s[i], r0, lower0, main0, upper0, B0, u0, Bu0, slopes0, Q_star0);
// 				g0 = gsl_ran_gaussian_pdf(rho0[obs_pos0] - obs, obs_sd);


// 				/* Weight computation */
// 				/* ------------------ */
// 				weights[i] = g0 * (double) res_signs[i] / (double) N0;
// 				absolute_weights[i] = fabs(weights[i]);
// 				normaliser += weights[i];
// 				abs_normaliser += absolute_weights[i];

// 				/* Reset the initial conditions to the current time level for the next particle weighting */
// 				memcpy(rho0, ics0, size0);

// 			}

// 		}



// 		/* --------------------------------------------------------------------------------------------------------- */
// 		/*																											 */
// 		/* Normalisation 																							 */
// 		/*																											 */
// 		/* --------------------------------------------------------------------------------------------------------- */
// 		ml_xhat = 0.0, sign_rat = 0.0;
// 		for (int i = 0; i < N_tot; i++) {
// 			absolute_weights[i] /= abs_normaliser;
// 			weights[i] /= normaliser;
// 			signs[i] = weights[i] < 0 ? -1 : 1;
// 			ml_weighted[n][i].x = s[i];
// 			ml_weighted[n][i].w = weights[i];
// 			ml_xhat += s[i] * weights[i];
// 		}
// 		xhats[n] = ml_xhat;
// 		// fprintf(XHATS, "%e ", ml_xhat);



// 		/* --------------------------------------------------------------------------------------------------------- */
// 		/*																											 */
// 		/* Resample and mutate 																						 */
// 		/*																											 */
// 		/* --------------------------------------------------------------------------------------------------------- */
// 		resample(N_tot, absolute_weights, ind, rng);
// 		for (int i = 0; i < N_tot; i++) {
// 			s_res[i] = s[ind[i]];
// 			res_signs[i] = signs[ind[i]];
// 		}
// 		mutate(N_tot, s, s_res, sig_sd, rng, n);

// 		if (n < lag) {

// 			/* Load the mutated particles into the historical particle array */
// 			for (int i = 0; i < N_tot; i++)
// 				X[i][n + 1] = s[i];

// 		}
// 		else {

// 			/* Shift the particles left one position and lose the oldest ancestor */
// 			for (int m = 0; m < lag; m++) {
// 				for (int i = 0; i < N_tot; i++)
// 					X[i][m] = X[i][m + 1];
// 			}

// 			/* Load the mutated particles into the vacant entry in the historical particle array */
// 			for (int i = 0; i < N_tot; i++)
// 				X[i][lag] = s[i];

// 			/* Evolve the fine initial condition with respect to the (n - lag)-th MSE-minimising point estimate */
// 			minmod_convection_diffusion_solver(rho1, nx1, dx1, dt1, obs_pos1, T_stop, CURVE_DATA, v, xhats[n - lag], r1, lower1, main1, upper1, B1, u1, Bu1, slopes1, Q_star1);

// 			/* Interpolate the coarse initial conditions from the fine initial condition evolved with respect to the nth MSE-minimising point estimate */
// 			gsl_interp_init(ics_interp, xs1, rho1, nx1 + 4);
// 			for (int j = 2; j < nx0 + 2; j++)
// 				rho0[j] = gsl_interp_eval(ics_interp, xs1, rho1, xs0[j], acc);

// 			memcpy(ics1, rho1, size1);
// 			memcpy(ics0, rho0, size0);

// 		}

// 	}
// 	if (N1 > 0)
// 		fprintf(L2_ERR_DATA, "%e ", l2_err);

// 	fclose(CURVE_DATA);
// 	// fclose(XHATS);
// 	fclose(CORRECTIONS);
// 	fclose(REGRESSION_DATA);
// 	fclose(EXACT_CORRECTIONS);

// 	free(signs);
// 	free(res_signs);
// 	free(ind);
// 	free(s);
// 	free(s_res);
// 	free(weights);
// 	free(absolute_weights);
// 	free(solns1);
// 	free(solns0);
// 	free(corrections);
// 	free(poly_weights);
// 	free(theta);
// 	free(xhats);
// 	free(rho0);
// 	free(rho1);
// 	free(ics0);
// 	free(ics1);
// 	free(slopes0);
// 	free(Q_star0);
// 	free(slopes1);
// 	free(Q_star1);
// 	free(xs0);
// 	free(xs1);
// 	free(PHI);
// 	free(C);
// 	free(C_inv);
// 	free(MP);
// 	free(X);

// 	gsl_matrix_free(C_gsl);
// 	gsl_permutation_free(p);
// 	gsl_matrix_free(C_inv_gsl);

// 	gsl_interp_free(ics_interp);
// 	gsl_interp_accel_free(acc);

// 	gsl_vector_free(lower0);
// 	gsl_vector_free(main0);
// 	gsl_vector_free(upper0);
// 	gsl_matrix_free(B0);
// 	gsl_vector_free(u0);
// 	gsl_vector_free(Bu0);
// 	gsl_vector_free(lower1);
// 	gsl_vector_free(main1);
// 	gsl_vector_free(upper1);	
// 	gsl_matrix_free(B1);
// 	gsl_vector_free(u1);
// 	gsl_vector_free(Bu1);

// }

