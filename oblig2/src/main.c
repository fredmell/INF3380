#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

struct Matrix{
  int num_rows, num_cols;
  double** array;
};

void init_matrix(struct Matrix *mat, int num_rows, int num_cols);
void read_matrix_bin(struct Matrix *mat, char *filename);
void write_matrix_bin(struct Matrix *mat, char *filename);
void read_cml(int argc, char **argv, char **input_fname, char **output_fname);
void free_matrix(struct Matrix *mat);
int isPerfectSquare(int p);
void checkNumProcs(int num_procs);
void distribute_matrix(double **my_a, double **whole_matrix, int m, int n,
  int my_m, int my_n, int procs_per_dim, int mycoords[2], MPI_Comm *comm_col, MPI_Comm *comm_row);

int main(int argc, char *argv[]) {
  int my_rank, num_procs, m, n, sqrt_p, my_m, my_n, rows, cols;
  int dims[2], periods[2], my2drank, mycoords[2];
  MPI_Comm comm_2d, comm_col, comm_row;
  struct Matrix A, B, C;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  if(my_rank == 0){
    // Allocate and load full matrices at process 0
    read_matrix_bin(&A, "../data/input/small_matrix_a.bin");
    read_matrix_bin(&B, "../data/input/small_matrix_b.bin");
    m = A.num_rows; n = A.num_cols;
    init_matrix(&C, A.num_rows, B.num_cols);
    checkNumProcs(num_procs); // Check that number of processes is a square number
    write_matrix_bin(&C, "../data/output/test.bin");
    free_matrix(&A); free_matrix(&B); free_matrix(&C);
  }
  // Broadcast multiplication dimensions
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  A.num_cols = m; A.num_rows = n;
  // Calculate partitioning
  // my_i = my_rank%sqrt_p; my_j = my_rank/sqrt_p;
  // rows = m/sqrt_p + (my_rank < m%sqrt_p);
  // cols = n/sqrt_p + (my_rank < n%sqrt_p);
  // int maxrows = m/sqrt_p + (m%sqrt_p > 0);
  // int maxcols = n/sqrt_p + (n%sqrt_p > 0);
  // printf("Proc %2d partition (%d, %d). I have %d rows, %d cols.\n", my_rank, my_j, my_i, rows, cols);
  // Allocate and broadcast local matrix blocks

  sqrt_p = (int) sqrt( (float) num_procs);
  // Set up MPI topology
  dims[0] = dims[1] = sqrt_p;
  periods[0] = periods[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

  MPI_Comm_rank(comm_2d, &my2drank);
  MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

  MPI_Cart_sub(comm_2d, (int[]){0, 1}, &comm_row);
  MPI_Cart_sub(comm_2d, (int[]){1, 0}, &comm_col);

  my_m = m/sqrt_p + (mycoords[0] < m % sqrt_p);
  my_n = n/sqrt_p + (mycoords[1] < n % sqrt_p);

  struct Matrix localA;
  init_matrix(&localA, my_m, my_n);

  distribute_matrix(localA.array, A.array, A.num_cols, A.num_rows, localA.num_cols, localA.num_rows, sqrt_p, mycoords, &comm_col, &comm_row);

  // int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
  MPI_Finalize();
  return 0;
}

// Read matrix file in binary format to 2D matrix with doubles
void read_matrix_bin(struct Matrix *mat, char* filename){
  int i;
  FILE* fp = fopen(filename,"rb");
  fread(&(mat->num_rows), sizeof(int), 1, fp);
  fread(&(mat->num_cols), sizeof(int), 1, fp);
  // Storage allocation of matrix
  mat->array = (double**)calloc(mat->num_rows, sizeof(double*));
  mat->array[0] = (double*)calloc((mat->num_rows*mat->num_cols), sizeof(double));
  for(i=1; i<mat->num_rows; i++){
    mat->array[i] = mat->array[i-1]+ mat->num_cols;
  }
  // Read entire matrix
  fread(mat->array[0], sizeof(double), (mat->num_rows*mat->num_cols), fp);
  fclose(fp);
}

void init_matrix(struct Matrix *mat, int num_rows, int num_cols){
  mat->num_rows = num_rows;
  mat->num_cols = num_cols;
  mat->array = (double**)calloc(mat->num_rows, sizeof(double*));
  mat->array[0] = (double*)calloc((mat->num_rows*mat->num_cols), sizeof(double));
  for(int i=1; i<mat->num_rows; i++){
    mat->array[i] = mat->array[i-1] + mat->num_cols;
  }
}

// Write 2D matrix with doubles to binary file
void write_matrix_bin(struct Matrix *mat, char *filename)
{
  FILE *fp = fopen(filename,"wb");
  fwrite(&(mat->num_rows), sizeof(int), 1, fp);
  fwrite(&(mat->num_cols), sizeof(int), 1, fp);
  fwrite(mat->array[0], sizeof(double), mat->num_rows*mat->num_cols, fp);
  fclose(fp);
}

// Read command line arguments and store filenames to read matrices from
void read_cml(int argc, char **argv, char **input_fname, char **output_fname){
  if(argc == 3)
  {
    *input_fname  = argv[1];
    *output_fname = argv[2];
  }
  else
  {
    printf("Invalid input. Need: input_filename output_filename\n");
  }
}

// Free dynamically allocated matrix memory
void free_matrix(struct Matrix *mat){
  free(mat->array[0]); // Free all entries
  free(mat->array);    // Free ptr
}

int isPerfectSquare(int p){
  double sr = sqrt( (double) p);
  return ((sr - floor(sr)) == 0);
}

void checkNumProcs(int num_procs){
  if(isPerfectSquare(num_procs) == 0){
    printf("Number of processes %d is not a perfect square and Cannon's algorithm will not work\n", num_procs);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void distribute_matrix(double **my_a, double **whole_matrix, int m, int n, int my_m, int my_n, int procs_per_dim, int mycoords[2], MPI_Comm *comm_col, MPI_Comm *comm_row)
{
    /* Buffers. */
    double *senddata_columnwise, *senddata_rowwise;

    /* Scatterv variables, step 1. */
    int *displs_y, *sendcounts_y, *everyones_m;

    /* Scatterv variables, step 2. */
    int *displs_x, *sendcounts_x, *everyones_n;
    MPI_Datatype columntype, columntype_scatter, columntype_recv, columntype_recv_scatter;

    displs_x = displs_y = sendcounts_x = sendcounts_y = everyones_m = everyones_n = NULL;
    senddata_columnwise = senddata_rowwise = NULL;

    if (mycoords[0] == 0 && mycoords[1] == 0)
    {
        senddata_columnwise = *whole_matrix;
    }

/* Step 1. Only first column participates. */
    if (mycoords[1] == 0)
    {
        if (mycoords[0] == 0)
        {
            everyones_m = (int *) calloc(procs_per_dim, sizeof(int));
            sendcounts_y = (int *)calloc(procs_per_dim, sizeof(int));
            displs_y = (int *)calloc(procs_per_dim + 1, sizeof(int));
        }

        MPI_Gather(&my_m, 1, MPI_INT, everyones_m, 1, MPI_INT, 0, *comm_col);

        if (mycoords[0] == 0)
        {
            displs_y[0] = 0;
            for (int i = 0; i < procs_per_dim; ++i)
            {
                sendcounts_y[i] = n * everyones_m[i];
                displs_y[i + 1] = displs_y[i] + sendcounts_y[i];
            }
        }

        senddata_rowwise = (double *) calloc(my_m * n, sizeof(double));

        MPI_Scatterv(senddata_columnwise, sendcounts_y, displs_y, MPI_DOUBLE, senddata_rowwise, my_m * n, MPI_DOUBLE, 0, *comm_col);
    }

    /* Step 2: Send data rowwise. */
    /* First, create the column data types. */
    MPI_Type_vector(my_m, 1, my_n, MPI_DOUBLE, &columntype);    /* Dummied out. */
    MPI_Type_commit(&columntype);
    MPI_Type_create_resized(columntype, 0, sizeof(double), &columntype_scatter);
    MPI_Type_commit(&columntype_scatter);

    /* Receivers need their own data type, or their data will be transposed! */
    MPI_Type_vector(my_m, 1, my_n, MPI_DOUBLE, &columntype_recv);
    MPI_Type_commit(&columntype_recv);
    MPI_Type_create_resized(columntype_recv, 0, sizeof(double), &columntype_recv_scatter);
    MPI_Type_commit(&columntype_recv_scatter);

    if (mycoords[1] == 0)
    {
        everyones_n = (int *) calloc(procs_per_dim, sizeof(int));
        sendcounts_x = (int *) calloc(procs_per_dim, sizeof(int));
        displs_x = (int *) calloc(procs_per_dim + 1, sizeof(int));
    }

    MPI_Gather(&my_n, 1, MPI_INT, everyones_n, 1, MPI_INT, 0, *comm_row);

    if (mycoords[1] == 0)
    {

        displs_x[0] = 0;
        for (int i = 0; i < procs_per_dim; ++i)
        {
            sendcounts_x[i] = everyones_n[i];
            displs_x[i + 1] = displs_x[i] + sendcounts_x[i];

        }
    }

    MPI_Scatterv(senddata_rowwise, sendcounts_x, displs_x, columntype_scatter, *my_a, my_n, columntype_recv_scatter, 0, *comm_row);

    /* And we have our matrices! */

    /* Finally, free everything. */
    MPI_Type_free(&columntype_recv_scatter);
    MPI_Type_free(&columntype_recv);

    if (mycoords[1] == 0)
    {
        free(displs_x);
        free(sendcounts_x);
        MPI_Type_free(&columntype_scatter);
        MPI_Type_free(&columntype);

        if (mycoords[0] == 0)
        {
            free(displs_y);
            free(sendcounts_y);
        }

        free(senddata_rowwise);
    }
}
