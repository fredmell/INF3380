#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stuff.h"

void allocate_image(image *u){
  u->image_data = calloc(u->m, sizeof(float*));
  for(int i = 0; i < u->m; i++){
    u->image_data[i] = calloc(u->n, sizeof(float));
  }
}

void deallocate_image(image *u){
  for(int i = u->m-1; i >= 0; i--){
    free(u->image_data[i]);
  }
  free(u->image_data);
}

void init_image(image *u, int m, int n){
  u->m = m;
  u->n = n;
  allocate_image(u);
}

void convert_jpeg_to_image(unsigned char *image_chars, image *u){
  for(int i = 0; i < u->m; i++){
    for(int j = 0; j < u->n; j++){
      u->image_data[i][j] = (float) image_chars[i*u->n + j];
    }
  }
}

void convert_image_to_jpeg(image *u, unsigned char *image_chars){
  for(int i = 0; i < u->m; i++){
    for(int j = 0; j < u->n; j++){
      image_chars[i*u->n + j] = (unsigned char) u->image_data[i][j];
    }
  }
}

void copy_array(image *u, image *u_bar){
  for(int i = 0; i < u->m; i++){
    for(int j = 0; j < u->n; j++){
      u_bar->image_data[i][j] = u->image_data[i][j];
    }
  }
}

void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters){
  // copy_array(u_bar, u);
	for(int k=0; k<iters; k++){
		for(int i=1; i < u->m-1; i++){
			for(int j=1; j<u->n-1; j++){
				u_bar->image_data[i][j] = u->image_data[i][j] +
					kappa*(u->image_data[i-1][j] + u->image_data[i][j-1] -
						   4*u->image_data[i][j] + u->image_data[i][j+1] +
						   u->image_data[i+1][j]);
			}
		}
    copy_array(u_bar, u);
	}
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

int main(int argc, char *argv[]){
  int m, n, c, iters;
  float kappa;
  image u, u_bar;
  unsigned char *image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;

  read_cml(argc, argv, &kappa, &iters, &input_jpeg_filename, &output_jpeg_filename);

  import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);

  init_image(&u, m, n);
  init_image(&u_bar, m, n);
  convert_jpeg_to_image(image_chars, &u);
  convert_jpeg_to_image(image_chars, &u_bar);
  iso_diffusion_denoising (&u, &u_bar, kappa, iters);

  convert_image_to_jpeg(&u, image_chars);
  export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);

  deallocate_image(&u);
  deallocate_image(&u_bar);
  return 0;
}
