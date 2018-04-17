#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

struct Matrix{
  char* filename;
  int num_rows, num_cols;
  double** array;
};

void read_matrix_bin(struct Matrix *mat);
void write_matrix_bin(struct Matrix *mat);
void read_cml(int argc, char **argv, char **input_fname, char **output_fname);
void free_matrix(struct Matrix *mat);

int main(int argc, char *argv[]) {
  int my_rank, num_procs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  if(my_rank == 0){
    struct Matrix A, B;
    A.filename = "../data/input/small_matrix_a.bin";
    B.filename = "../data/input/small_matrix_b.bin";
    read_matrix_bin(&A);
    read_matrix_bin(&B);
    free_matrix(&A);
    free_matrix(&B);
  }
  MPI_Finalize();
  return 0;
}

// Read matrix file in binary format to 2D matrix with doubles
void read_matrix_bin(struct Matrix *mat){
  int i;
  FILE* fp = fopen(mat->filename,"rb");
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

// Write 2D matrix with doubles to binary file
void write_matrix_bin(struct Matrix *mat)
{
  FILE *fp = fopen(mat->filename,"wb");
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
