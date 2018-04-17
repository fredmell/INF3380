#include "stuff.h"

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
};
