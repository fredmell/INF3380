#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

/*
Multiply two matrices using Cannon's algorithm. Output is written to .bin file.
*/

struct Matrix{
  int num_rows, num_cols;
  double** array;
};

void init_matrix(struct Matrix *mat, int num_rows, int num_cols);
void read_matrix_bin(struct Matrix *mat, char *filename);
void write_matrix_bin(struct Matrix *mat, char *filename);
void read_cml(int argc, char **argv, char **, char **, char **);
void free_matrix(struct Matrix *mat);
int isPerfectSquare(int p);
void checkNumProcs(int num_procs);
void distribute_matrix(double **my_a, double **whole_matrix, int m, int n,
  int my_m, int my_n, int procs_per_dim, int mycoords[2], MPI_Comm *comm_col, MPI_Comm *comm_row);
void gather_matrix(double **my_mat, double **whole_mat, int m, int n,
  int my_m, int my_n, int procs_per_dim, int mycoords[2], MPI_Comm *comm_col, MPI_Comm *comm_row);
void MatrixMultiply(struct Matrix *A, struct Matrix *B, struct Matrix *C, int, int, int);

int main(int argc, char *argv[]) {
  int my_rank, num_procs, m, n, sqrt_p, my_m, my_n, rows, cols, shiftsource, shiftdest;
  int dims[2], periods[2], my2drank, mycoords[2], uprank, downrank, leftrank, rightrank;
  MPI_Comm comm_2d, comm_col, comm_row;
  MPI_Status status;
  struct Matrix A, B, C;
  char *outfile;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  if(my_rank == 0){
    // Read filenames from run time cml arguments
    char *f1, *f2;
    read_cml(argc, argv, &f1, &f2, &outfile);
    // Allocate and load full matrices at process 0
    read_matrix_bin(&A, f1);
    read_matrix_bin(&B, f2);
    checkNumProcs(num_procs); // Check that number of processes is a square number;
  }
  // Broadcast multiplication dimensions
  MPI_Bcast(&A.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&A.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  m = A.num_rows; n = B.num_cols;
  int l = A.num_cols;

  sqrt_p = (int) sqrt( (float) num_procs); // Square root of number of processes = Procs per dim

  // Set up MPI Cartesian topology
  dims[0] = dims[1] = sqrt_p;
  periods[0] = periods[1] = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_2d);

  MPI_Comm_rank(comm_2d, &my2drank);
  MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

  // Creat subtopologies for distribution and gathering
  MPI_Cart_sub(comm_2d, (int[]){0, 1}, &comm_row);
  MPI_Cart_sub(comm_2d, (int[]){1, 0}, &comm_col);

  // Calculate local matrix dimensions
  int my_m_a = m/sqrt_p + (mycoords[0] < (m%sqrt_p));
  int my_n_a = l/sqrt_p + (mycoords[1] < (l%sqrt_p));

  int my_m_b = l/sqrt_p + (mycoords[0] < (l%sqrt_p));
  int my_n_b = n/sqrt_p + (mycoords[1] < (n%sqrt_p));

  struct Matrix tmpA, tmpB, localA, localB, localC;

  // Find maximum local matrix dimensions and distribute. Possible improvement: Only maximum along rows in A and columns in B.
  int max_l_a, max_l_b, max_m_a, max_n_b;
  if(mycoords[0]==0 && mycoords[1]==0) {max_l_a = my_n_a; max_l_b = my_m_b; max_m_a = my_m_a, max_n_b = my_n_b;}
  MPI_Bcast(&max_l_a, 1, MPI_INT, 0, comm_2d);
  MPI_Bcast(&max_l_b, 1, MPI_INT, 0, comm_2d);
  MPI_Bcast(&max_m_a, 1, MPI_INT, 0, comm_2d);
  MPI_Bcast(&max_n_b, 1, MPI_INT, 0, comm_2d);

  // Allocate temporary local matrices with correct dimensions and also local matrices with maximum size for easy sendrecv
  init_matrix(&tmpA, my_m_a, my_n_a); init_matrix(&localA, max_m_a, max_l_a);
  init_matrix(&tmpB, my_m_b, my_n_b); init_matrix(&localB, max_l_b, max_n_b);
  init_matrix(&localC, max_m_a, max_n_b);

  // Distribute A to local tmp matrices
  distribute_matrix(tmpA.array, A.array, m, l, my_m_a, my_n_a, sqrt_p, mycoords, &comm_col, &comm_row);
  distribute_matrix(tmpB.array, B.array, l, n, my_m_b, my_n_b, sqrt_p, mycoords, &comm_col, &comm_row);

  // Transfer values from tmp matrices to large local matrices
  for(int i=0; i<my_m_a; i++){
    for(int j=0; j<my_n_a; j++){
      localA.array[i][j] = tmpA.array[i][j];
    }
  }
  for(int i=0; i<my_m_b; i++){
    for(int j=0; j<my_n_b; j++){
      localB.array[i][j] = tmpB.array[i][j];
    }
  }
  free_matrix(&tmpA); // Free temp matrices
  free_matrix(&tmpB);

  // Calculate ranks for shifting during iterations
  MPI_Cart_shift(comm_2d, 1, -1, &rightrank, &leftrank);
  MPI_Cart_shift(comm_2d, 0, -1, &downrank, &uprank);

  int sizeLocalA = max_m_a*max_l_a; // Local matrix sizes to avoid repeating same multiplication
  int sizeLocalB = max_l_b*max_n_b; // Should be equal

  // Save dimensions of C for later
  int Cdims[2];
  Cdims[0] = my_m_a;
  Cdims[1] = my_n_b;

  // Initial alignment of A (shift down by y-coordinate). Also update local dimensions.
  MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace(*localA.array, sizeLocalA, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_2d, &status);
  MPI_Sendrecv_replace(&my_m_a, 1, MPI_INT, shiftdest, 1, shiftsource, 1, comm_2d, &status);
  MPI_Sendrecv_replace(&my_n_a, 1, MPI_INT, shiftdest, 1, shiftsource, 1, comm_2d, &status);

  // Initial alignment of B (shift up by x-coordinate). Also update local dimensions.
  MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace(*localB.array, sizeLocalB, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_2d, &status);
  MPI_Sendrecv_replace(&my_n_b, 1, MPI_INT, shiftdest, 1, shiftsource, 1, comm_2d, &status);

  // Perform Cannon's algo loop
  for(int i=0; i<sqrt_p; i++){
    MatrixMultiply(&localA, &localB, &localC, my_m_a, my_n_b, my_n_a);

    MPI_Sendrecv_replace(*localA.array, sizeLocalA, MPI_DOUBLE, leftrank, 1, rightrank, 1, comm_2d, &status);
    MPI_Sendrecv_replace(&my_m_a, 1, MPI_INT, leftrank, 1, rightrank, 1, comm_2d, &status);
    MPI_Sendrecv_replace(&my_n_a, 1, MPI_INT, leftrank, 1, rightrank, 1, comm_2d, &status);

    MPI_Sendrecv_replace(*localB.array, sizeLocalB, MPI_DOUBLE, uprank, 1, downrank, 1, comm_2d, &status);
    MPI_Sendrecv_replace(&my_n_b, 1, MPI_INT, uprank, 1, downrank, 1, comm_2d, &status);
  }

  struct Matrix tmpC;
  init_matrix(&tmpC, Cdims[0], Cdims[1]);
  for(int i=0; i<Cdims[0]; i++){
    for (int j=0; j<Cdims[1]; j++){
      tmpC.array[i][j] = localC.array[i][j];
    }
  }

  // Allocate and gather full C matrix and write to file.
  if(mycoords[0] == 0 && mycoords[1] == 0) init_matrix(&C, m, n);
  gather_matrix(tmpC.array, C.array, m, n, Cdims[0], Cdims[1], sqrt_p, mycoords, &comm_col, &comm_row);
  if(mycoords[0] == 0 && mycoords[1] == 0) write_matrix_bin(&C, outfile);

  // Free all matrices
  if(mycoords[0]==0 && mycoords[1]==0){
    free_matrix(&A); free_matrix(&B); free_matrix(&C);
  }
  free_matrix(&localA); free_matrix(&localB); free_matrix(&localC);
  free_matrix(&tmpC);
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
  printf("Successfully wrote multiplication result to: %s\n", filename);
}

// Read command line arguments and store filenames to read matrices from
void read_cml(int argc, char **argv, char **input_fname1, char **input_fname2, char **output_fname){
  if(argc == 1){
    printf("Defaulted to small matrices. For other matrices add input filenames and output filename as arguments.\n");
    *input_fname1 = "../data/input/small_matrix_a.bin";
    *input_fname2 = "../data/input/small_matrix_b.bin";
    *output_fname = "../data/output/small_matrix_c.bin";
  }
  else if(argc == 2){
    if(strcmp(argv[1], "large") == 0){
      printf("Using large example matrices.\n");
      *input_fname1 = "../data/input/large_matrix_a.bin";
      *input_fname2 = "../data/input/large_matrix_b.bin";
      *output_fname = "../data/output/large_matrix_c.bin";
    }
    else if(strcmp(argv[1], "small") == 0){
      printf("Using small example matrices.\n");
      *input_fname1 = "../data/input/small_matrix_a.bin";
      *input_fname2 = "../data/input/small_matrix_b.bin";
      *output_fname = "../data/output/small_matrix_c.bin";
    }
    else{
      printf("Invalid input. Valid example matrices are used with arguments large and small. For other matrices enter filenames and outfilename.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  else if(argc == 4){
    *input_fname1  = argv[1];
    *input_fname2 = argv[2];
    *output_fname = argv[3];
  }
  else
  {
    printf("Invalid input. Possible options are:\n   input_filename1 input_filename2 output_filename \nOR\n   small/large (uses example matrices)\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
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
    MPI_Type_vector(my_m, 1, n, MPI_DOUBLE, &columntype);    /* Dummied out. */
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

void gather_matrix(double **my_mat, double **whole_mat, int m, int n, int my_m, int my_n, int procs_per_dim, int mycoords[2], MPI_Comm *comm_col, MPI_Comm *comm_row){
  // Buffers
  double *recvdata_columnwise, *recvdata_rowwise;

  // Gatherv variables
  int *displs_y, *recvcounts_y, *everyones_m;

  int *displs_x, *recvcounts_x, *everyones_n;
  MPI_Datatype columntype, columntype_gather, columntype_send, columntype_send_gather;

  displs_x = displs_y = recvcounts_x = recvcounts_y = everyones_m = everyones_n = NULL;
  recvdata_columnwise = recvdata_rowwise = NULL;

  // Allocate stuff for both steps
  if (mycoords[1] == 0)
  {
      if (mycoords[0] == 0)
      {
          everyones_m = (int *) calloc(procs_per_dim, sizeof(int));
          recvcounts_y = (int *)calloc(procs_per_dim, sizeof(int));
          displs_y = (int *)calloc(procs_per_dim + 1, sizeof(int));
      }

      MPI_Gather(&my_m, 1, MPI_INT, everyones_m, 1, MPI_INT, 0, *comm_col);

      if (mycoords[0] == 0)
      {
          displs_y[0] = 0;
          for (int i = 0; i < procs_per_dim; ++i)
          {
              recvcounts_y[i] = n * everyones_m[i];
              displs_y[i + 1] = displs_y[i] + recvcounts_y[i];
          }
      }

      recvdata_rowwise = (double *) calloc(my_m * n, sizeof(double));
  }

  if (mycoords[1] == 0)
  {
      everyones_n = (int *) calloc(procs_per_dim, sizeof(int));
      recvcounts_x = (int *) calloc(procs_per_dim, sizeof(int));
      displs_x = (int *) calloc(procs_per_dim + 1, sizeof(int));
  }

  MPI_Gather(&my_n, 1, MPI_INT, everyones_n, 1, MPI_INT, 0, *comm_row);

  if (mycoords[1] == 0)
  {

      displs_x[0] = 0;
      for (int i = 0; i < procs_per_dim; ++i)
      {
          recvcounts_x[i] = everyones_n[i];
          displs_x[i + 1] = displs_x[i] + recvcounts_x[i];

      }
  }

  MPI_Type_vector(my_m, 1, n, MPI_DOUBLE, &columntype);    /* Dummied out. */
  MPI_Type_commit(&columntype);
  MPI_Type_create_resized(columntype, 0, sizeof(double), &columntype_gather);
  MPI_Type_commit(&columntype_gather);

  /* Receivers need their own data type, or their data will be transposed! */
  MPI_Type_vector(my_m, 1, my_n, MPI_DOUBLE, &columntype_send);
  MPI_Type_commit(&columntype_send);
  MPI_Type_create_resized(columntype_send, 0, sizeof(double), &columntype_send_gather);
  MPI_Type_commit(&columntype_send_gather);

  MPI_Gatherv(*my_mat, my_n, columntype_send_gather, recvdata_rowwise, recvcounts_x, displs_x, columntype_gather, 0, *comm_row);

  if (mycoords[0] == 0 && mycoords[1] == 0)
  {
      recvdata_columnwise = *whole_mat;
  }

  if (mycoords[1] == 0){
    MPI_Gatherv(recvdata_rowwise, my_m*n, MPI_DOUBLE, recvdata_columnwise, recvcounts_y, displs_y, MPI_DOUBLE, 0, *comm_col);
  }

  // Free everything
  MPI_Type_free(&columntype_send_gather);
  MPI_Type_free(&columntype_send);

  if (mycoords[1] == 0){
    free(displs_x);
    free(recvcounts_x);
    MPI_Type_free(&columntype_gather);
    MPI_Type_free(&columntype);

    if (mycoords[0] == 0)
    {
        free(displs_y);
        free(recvcounts_y);
    }

    free(recvdata_rowwise);
  }
}

void MatrixMultiply(struct Matrix *A, struct Matrix *B, struct Matrix *C, int m, int n, int l){
  // Multiply matrices A (m x l) and B (l x n) to get C (m x n). Uses user provided dimensions
  // because some local matrices have more memory allocated than their actual dimensions.

  // Static scheduling because each iteration should take approx. same amount of time
  #pragma omp parallel for schedule(static)
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      for(int k=0; k<l; k++){
        C->array[i][j] += A->array[i][k] * B->array[k][j];
      }
    }
  }
}
