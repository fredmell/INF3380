#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "../serial/stuff.h"

void read_cml(int argc, char **argv, float *kappa, int *iters, char **input_fname, char **output_fname);
void allocate_image(image *whole_image, int m, int n);
void deallocate_image(image *u, int m);
void convert_jpeg_to_image(unsigned char *image_chars, image *u, int m, int n);
void convert_image_to_jpeg(image *u, unsigned char *image_chars, int m, int n);
void copy_array(image *u, image *u_bar, int, int);
void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa, int iters, int, int, int, int);

int main (int argc, char *argv[]) {
  int m, n, c, iters;
  int my_n, my_rank, num_procs;
  float kappa;
  image u, u_bar, whole_image;
  unsigned char *image_chars, *my_image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);

  read_cml(argc, argv, &kappa, &iters, &input_jpeg_filename, &output_jpeg_filename);

  if (my_rank==0) {
    import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    allocate_image (&whole_image, m, n);
  }
  MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // /* divide the m x n pixels evenly among the MPI processes */
  my_n = n;
  int rem = m%num_procs;

  int sum = 0;
  int *sendcounts = calloc(sizeof(int), num_procs);
  int *displs = calloc(sizeof(int), num_procs);
  int *local_ms = calloc(sizeof(int), num_procs);

  // Calculate counts and displacements
  for(int i = 0; i < num_procs; i++){
    local_ms[i] = m/num_procs;
    if(i < rem){
      local_ms[i]++;
    }
    if(i == 0){
      local_ms[i]++;
    }

    if(i > 0 && i < num_procs-1){
      local_ms[i] += 2;
      sum -= 2*n;
    }

    if(i == num_procs-1){
      local_ms[i]++;
      sum -= 2*n;
    }

    sendcounts[i] = local_ms[i]*n;
    displs[i] = sum;
    sum+=sendcounts[i];
  }


  // print calculated send counts and displacements for each process
   if (0 == my_rank) {
       for (int i = 0; i < num_procs; i++) {
           printf("sendcounts[%d]/n = %d\tdispls[%d]/n = %d\n", i, sendcounts[i]/n, i, displs[i]/n);
       }
   }
   printf("I am process %d and local_ms[my_rank] is %d\n", my_rank, local_ms[my_rank]);

  allocate_image(&u, local_ms[my_rank], my_n);
  allocate_image(&u_bar, local_ms[my_rank], my_n);

  my_image_chars = calloc(local_ms[my_rank]*n, sizeof(unsigned char));
  MPI_Scatterv(image_chars, sendcounts, displs, MPI_UNSIGNED_CHAR, my_image_chars, local_ms[my_rank]*n,
              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  convert_jpeg_to_image (my_image_chars, &u, local_ms[my_rank], my_n);
  convert_jpeg_to_image (my_image_chars, &u_bar, local_ms[my_rank], my_n);
  iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters, local_ms[my_rank], my_n, my_rank, num_procs);
  convert_image_to_jpeg(&u_bar, my_image_chars, local_ms[my_rank], my_n);

  MPI_Gatherv(my_image_chars, local_ms[my_rank]*n, MPI_UNSIGNED_CHAR, image_chars, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

  if (my_rank==0) {
    // convert_image_to_jpeg(&whole_image, image_chars, m, n);
    export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
    deallocate_image (&whole_image, m);
  }
  deallocate_image (&u, local_ms[my_rank]);
  deallocate_image (&u_bar, local_ms[my_rank]);
  free(sendcounts); free(displs); free(local_ms);
  MPI_Finalize ();
  return 0;
}

void read_cml(int argc, char **argv, float *kappa, int *iters,
              char **input_fname, char **output_fname){
  if(argc == 5)
  {
    *kappa        = atof(argv[1]);
    *iters        = atoi(argv[2]);
    *input_fname  = argv[3];
    *output_fname = argv[4];
  }
  else
  {
    printf("Wrong input. Need: kappa iters input_filename out_filename\n");
  }
}

void allocate_image(image *u, int m, int n){
  u->image_data = calloc(m, sizeof(float*));
  for(int i = 0; i < m; i++){
    u->image_data[i] = calloc(n, sizeof(float));
  }
}

void deallocate_image(image *u, int m){
  for(int i = m-1; i >= 0; i--){
    free(u->image_data[i]);
  }
  free(u->image_data);
}

void convert_jpeg_to_image(unsigned char *image_chars, image *u, int m, int n){
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      u->image_data[i][j] = (float) image_chars[i*n + j];
    }
  }
}

void copy_array(image *u, image *u_bar, int m, int n){
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      u_bar->image_data[i][j] = u->image_data[i][j];
    }
  }
}

void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa, int iters, int m, int n, int rank, int num_procs){
  // copy_array(u_bar, u);
	for(int k=0; k<iters; k++){
		for(int i=1; i < m-1; i++){
			for(int j=1; j< n-1; j++){
				u_bar->image_data[i][j] = u->image_data[i][j] +
					kappa*(u->image_data[i-1][j] + u->image_data[i][j-1] -
						   4*u->image_data[i][j] + u->image_data[i][j+1] +
						   u->image_data[i+1][j]);
			}
		}
    // Exchange rows with next process (except for the last)
    if(rank!=num_procs-1){
      printf("k = %d ... rank = %d\n", k, rank);
      MPI_Send(u_bar->image_data[m-2], n, MPI_FLOAT, rank+1, k, MPI_COMM_WORLD);
      MPI_Recv(u_bar->image_data[m-1], n, MPI_FLOAT, rank+1, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // Exchange rows with preceeding process (except for the first)
    if(rank != 0){
      printf("k = %d ... rank = %d\n", k, rank);
      MPI_Recv(u_bar->image_data[0], n, MPI_FLOAT, rank-1, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(u_bar->image_data[1], n, MPI_FLOAT, rank-1, k, MPI_COMM_WORLD);
    }
    copy_array(u_bar, u, m, n);
	}
}

void convert_image_to_jpeg(image *u, unsigned char *image_chars, int m, int n){
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      image_chars[i*n + j] = (unsigned char) u->image_data[i][j];
    }
  }
}
