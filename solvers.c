// #include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "solvers.h"

const double CFL_CONST2 = 0.8;

void generate_ics(double * ics, double dx, int nx, double space_left) {

	/** Construct the interior of the initial conditions */
	double x = space_left;
	for (int j = 0; j < nx + 1; j++) {
		ics[j] = 0.0;
		x += dx;
	}
}


double minmod(double a, double b) {

	if ( (fabs(a) < fabs(b)) && (a * b > 0) )
		return a;
	if ( (fabs(b) < fabs(a)) && (a * b > 0) )
		return b;
	return 0.0;

}


void construct_forward(double r, int nx, gsl_vector * lower, gsl_vector * main, gsl_vector * upper) {

	/* We only deal with the sub-diagonals which do not depend on the source term */
	for (int n = 0; n < nx - 1; n++) {
		lower->data[n] = -r;
		upper->data[n] = -r;
	}
	lower->data[nx - 2] = 0.0;
	upper->data[0] = 0.0;

}


void construct_present(double r, int nx, gsl_matrix * B) {

	/* We only deal with the sub-diagonals which do not depend on the source term */
	for (int n = 1; n < nx; n++)
		gsl_matrix_set(B, n, n - 1, r);
	gsl_matrix_set(B, nx - 1, nx - 2, 0.0);

	for (int n = 0; n < nx - 1; n++)
		gsl_matrix_set(B, n, n + 1, r);
	gsl_matrix_set(B, 0, 1, 0.0);

}


void crank_nicolson(int nx, double dt, double T_stop, gsl_vector * lower, gsl_vector * main, gsl_vector * upper, gsl_matrix * B, gsl_vector * u, gsl_vector * Bu) {

	/* Fine v such that Av = Bu and set v <- u */
	double t = 0.0;
	while (t < T_stop) {
		gsl_blas_dgemv(CblasNoTrans, 1.0, B, u, 0.0, Bu);
		gsl_linalg_solve_tridiag(main, upper, lower, Bu, u);
		t += dt;
	}
}


void minmod_convection_diffusion_solver(double * rho, int nx, double dx, double dt, int obs_pos, double T_stop, FILE * CURVE_DATA, double v, double s, double r, gsl_vector * lower, gsl_vector * main, gsl_vector * upper, gsl_matrix * B, gsl_vector * u, gsl_vector * Bu, double * slopes, double * Q_star) {

	int time_counter = 0;
	double mid_plus = 1 + 2 * r + s * dt;
	double mid_minus = 1 - 2 * r - s * dt;
	// double dt_fine = dt / 2.0;
	double dt_fine = dt / 1.0;
	double t = 0.0, exp_scaler = 0.05;
	double f_step = 0.5 * CFL_CONST2 * (dx - v * dt);

	/* Set the particle-dependent matrix solver entries */
	for (int n = 1; n < nx + 1; n++) {
		main->data[n] = mid_plus;
		gsl_matrix_set(B, n, n, mid_minus);
	}
	main->data[0] = 1.0, main->data[nx + 1] = 1.0;
	gsl_matrix_set(B, 0, 0, 1.0); 
	gsl_matrix_set(B, nx + 1, nx + 1, 1.0);

	/* Solve over the integration time */
	while (t < T_stop) {

		/* Set the values of the ghost cells */
		rho[nx + 2] = rho[nx + 1];
		rho[nx + 3] = rho[nx + 2];
		rho[0] = exp(-t / exp_scaler) + 1;
		// rho[0] = sin(s * t / T_stop) + 2.0;
		rho[1] = rho[0];


		/* A-step */
		/* ------ */
		/* Compute the limiter terms */
		for (int j = 1; j < nx + 2; j++)			
			slopes[j - 1] = minmod((rho[j] - rho[j - 1]) / dx, (rho[j + 1] - rho[j]) / dx);

		/* Update using the slope-limited conservative formula */
		for (int j = 2; j < nx + 2; j++)
			Q_star[j - 2] = rho[j] - CFL_CONST2 * (rho[j] - rho[j - 1]) - f_step * (slopes[j - 1] - slopes[j - 2]);


		/* B-step */
		/* ------ */
		/* Solve the heat equation (diffusive) part of the problem */
		u->data[0] = rho[1];
		u->data[nx + 1] = rho[nx + 2];
		for (int j = 1; j < nx + 1; j++)
			u->data[j] = Q_star[j - 1];
		crank_nicolson(nx + 2, dt_fine, dt, lower, main, upper, B, u, Bu);


		/* Update solution */
		/* --------------- */
		for (int j = 2; j < nx + 2; j++)
			rho[j] = u->data[j - 1];
		t += dt;
		// time_counter++;

		// for (int j = 2; j < nx + 2; j++)
			// fprintf(CURVE_DATA, "%e ", rho[j]);

	}
	// fprintf(CURVE_DATA, "\n");
	// fprintf(CURVE_DATA, "%d\n", time_counter);

}


void construct_present_mat(gsl_matrix * B, int nx, double c, double d, double r) {

	/** Assuming matrix is of dim (nx + 2) x (nx + 2) with equal ghost cells for the solution */

	/* Sub-diagonal */
	/* ------------ */
	for (int n = 1; n < nx + 1; n++)
		gsl_matrix_set(B, n, n - 1, c);
	gsl_matrix_set(B, nx + 1, nx, 0);
	

	/* Main-diagonal */
	/* ------------- */
	for (int n = 1; n < nx + 1; n++)
		gsl_matrix_set(B, n, n, d);
	gsl_matrix_set(B, 0, 0, 1);
	gsl_matrix_set(B, nx + 1, nx + 1, 1);


	/* Sup-diagonal */
	/* ------------ */
	for (int n = 1; n < nx + 1; n++)
		gsl_matrix_set(B, n, n + 1, r);
	gsl_matrix_set(B, 0, 1, 0);

}


void construct_forward_mat(gsl_vector * main, gsl_vector * upper, gsl_vector * lower, int nx, double a, double b, double r) {

	/** Assuming matrix is of dim (nx + 2) x (nx + 2) with ghost cells equal to the present time ghost cells */

	/* Sub-diagonal */
	/* ------------ */
	for (int n = 0; n < nx; n++)
		gsl_vector_set(lower, n, a);
	gsl_vector_set(lower, nx, 0);


	/* Main-diagonal */
	/* ------------- */
	for (int n = 1; n < nx + 1; n++)
		gsl_vector_set(main, n, b);
	gsl_vector_set(main, 0, 1);
	gsl_vector_set(main, nx + 1, 1);


	/* Sup-diagonal */
	/* ------------ */
	for (int n = 1; n < nx + 1; n++)
		gsl_vector_set(upper, n, -r);
	gsl_vector_set(upper, 0, 0);

}


void solve(int nx, int nt, double dx, double dt, gsl_matrix * B, gsl_vector * rho, gsl_vector * rho_tilde, double s, double rdx_sq, gsl_vector * main, gsl_vector * upper, gsl_vector * lower, FILE * CURVE_DATA) {

	/* Implement the equation Av+ = Bv */
	/* ------------------------------- */
	for (int t = 0; t < nt; t++) {

		/* Set the boundary conditions */
		rho->data[0] = fabs(s) * exp(-t / (double) nt); ////////
		rho->data[nx + 1] = rho->data[nx];

		/* Construct the RHS */
		gsl_blas_dgemv(CblasNoTrans, 1.0, B, rho, 0.0, rho_tilde);
		for (int n = 0; n < nx + 2; n++) {
			rho_tilde->data[n] += 2 * rdx_sq * s * s * exp(s * (1 - n * dx)); // s is constant case
		}

		/* Solve for the LHS */
		gsl_linalg_solve_tridiag(main, upper, lower, rho_tilde, rho);

		/* Print the solution */
		// for (int j = 1; j < nx + 1; j++)
		// 	fprintf(CURVE_DATA, "%e ", rho->data[j]);
		// fprintf(CURVE_DATA, "\n");

	}

}

















