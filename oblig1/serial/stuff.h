#ifndef STUFF_H
#define STUFF_H

typedef struct {

    int m, n, c;
    float **image_data;

} image;

// Import and export functions are in simple-jpeg
void import_JPEG_file(const char *filename, unsigned char **image_chars,
                      int *image_height, int *image_width, int *num_components);
void export_JPEG_file(const char *filename, unsigned char *image_chars,
                      int image_height, int image_width, int num_components,
                      int quality);

#endif // STUFF_H
