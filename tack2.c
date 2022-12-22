#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define  max(a, b) ((a) > (b) ? (a) : (b))

#define EPS_MAX .1e-7
#define IT_MAX 100

static int rank;
static int size;
static int err_occured;

MPI_Comm main_comm;

static void err_handler(MPI_Comm *pcomm, int *perr, ...) {
    char errstr[MPI_MAX_ERROR_STRING];
    int size, nf, len;
    MPI_Group group_f;

    MPI_Comm_size(main_comm, &size);
    MPIX_Comm_failure_ack(main_comm);
    MPIX_Comm_failure_get_acked(main_comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(*perr, errstr, &len);

    printf("\nRank %d / %d: Notified of error %s. %d found dead\n", rank, size, errstr, nf);
	err_occured = 1;

    MPIX_Comm_shrink(main_comm, &main_comm);
    MPI_Comm_rank(main_comm, &rank);
}

void *new_array(int N)
{
	double **A = (double **)malloc(N * sizeof(double *));
	for (int i = 0; i < N; ++i)
	{
		A[i] = (double *)malloc(N * sizeof(double));
		memset(A[i], 0, N * sizeof(double));	
	}
	
	return A;
}

void init(double **A, int N)
{ 
checkpoint:

	if (rank == size)
		goto spare_proc;

	int left = 1 + (N - 2) * rank / size;
    int right = 1 + (N - 2) * (rank + 1) / size;

	for(int i = left; i < right; ++i)
		for(int j = 1; j < N - 1; ++j)
				A[i][j] = ( 1 + i + j );

spare_proc:
	MPI_Barrier(main_comm);

	if (err_occured)
	{
		err_occured = 0;
		goto checkpoint;
	}
}

void pairwise_exchange(double **A, int N)
{
	static double *new_left = NULL;
	static double *new_right = NULL;

	if (!new_left)
		new_left = (double *)malloc(N * sizeof(double));
	if (!new_right)
		new_right = (double *)malloc(N * sizeof(double));

	memset(new_left, 0, N * sizeof(double));
	memset(new_right, 0, N * sizeof(double));

	int left = 1 + (N - 2) * rank / size;
    int right = 1 + (N - 2) * (rank + 1) / size;

	MPI_Status status;
	MPI_Request request;

	if (rank != 0)
		MPI_Isend(A[left], N, MPI_DOUBLE, rank - 1, 0, main_comm, &request);

	if (rank != size - 1)
		MPI_Isend(A[right - 1], N, MPI_DOUBLE, rank + 1, 0, main_comm, &request);


	if (rank != 0)
		MPI_Recv(new_left, N, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, main_comm, &status);

	if (rank != size - 1)
		MPI_Recv(new_right, N, MPI_DOUBLE, rank + 1, MPI_ANY_TAG, main_comm, &status);

	memmove(A[left - 1], new_left, N * sizeof(double));
	memmove(A[right], new_right, N * sizeof(double));
}

void relax(double **A, double **B, int N)
{
checkpoint:

	if (rank == size)
		goto spare_proc;

	pairwise_exchange(A, N);

	int left = 1 + (N - 2) * rank / size;
    int right = 1 + (N - 2) * (rank + 1) / size;

	for(int i = left; i < right; ++i)
		for(int j = 1; j < N - 1; ++j)
		{
			B[i][j] = (A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]) / 4;
		}

spare_proc:
	MPI_Barrier(main_comm);

	if (err_occured)
	{
		err_occured = 0;
		goto checkpoint;
	}
}

void resid(double **A, double **B, int N, double *eps)
{ 
checkpoint:

	if (rank == size)
		goto spare_proc;

	int left = 1 + (N - 2) * rank / size;
    int right = 1 + (N - 2) * (rank + 1) / size;

	for(int i = left; i < right; ++i)
		for(int j = 1; j < N - 1; ++j)
		{
			double new_eps = fabs(A[i][j] - B[i][j]);
			*eps = max(*eps, new_eps);
			A[i][j] = B[i][j]; 
		}

spare_proc:
	MPI_Barrier(main_comm);

	if (err_occured)
	{
		err_occured = 0;
		goto checkpoint;
	}

	double all_eps;
	MPI_Allreduce(eps, &all_eps, 1, MPI_DOUBLE, MPI_MAX, main_comm);
	*eps = all_eps;
}

void verify(double **A, int N)
{
checkpoint:

	if (rank == size)
		goto spare_proc;

	int left = 1 + (N - 2) * rank / size;
    int right = 1 + (N - 2) * (rank + 1) / size;

	double sum = 0;

	for (int i = 0; i < N; ++i)
		for (int j = 0; j < N; ++j)
		{
			sum += A[i][j] * (i + 1) * (j + 1) / (N * N);
		}

spare_proc:
	MPI_Barrier(main_comm);

	if (err_occured)
	{
		err_occured = 0;
		goto checkpoint;
	}

	double all_sum;
	MPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, main_comm);

	if (rank == 0)
		printf("  S = %lf\n", all_sum);
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		if (rank == 0)
			printf("No size specified\n");
		
		MPI_Finalize();
		return 0;
	}

	MPI_Init(&argc, &argv);
	main_comm = MPI_COMM_WORLD;

	MPI_Comm_size(main_comm, &size);
    MPI_Comm_rank(main_comm, &rank);

	MPI_Errhandler errh;
    MPI_Comm_create_errhandler(err_handler, &errh);
    MPI_Comm_set_errhandler(main_comm, errh);
    MPI_Barrier(main_comm);

	size--;

	int N;
	sscanf(argv[1], "%d", &N);

	double **A = new_array(N);
	double **B = new_array(N);

	init(A, N);

	for(int it = 1; it <= IT_MAX; ++it)
	{
		double eps = 0.;
		
		relax(A, B, N);
		resid(A, B, N, &eps);
		
		printf( "it=%4i   eps=%f\n", it, eps);
		if (eps < EPS_MAX) 
			break;
	}

	MPI_Barrier(main_comm);
	verify(A, N);

	MPI_Finalize();
	return 0;
}