/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__
void shmem_reduce_kernel(float * d_max_out, float * d_min_out, const float * d_in) {
  
  assert(blockDim.x % 2 == 0);
  // according to stackoverflow, extern __shared__ array can only have one copy
  // in order to fulfill the need of 2 array, double the size
  extern __shared__ float max_min_data[];
  
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_in_block = threadIdx.x;
  
  max_min_data[idx_in_block] = d_in[index];
  max_min_data[idx_in_block + blockDim.x] = d_in[index]
  __syncthreads();
  
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (idx_in_block < s) {
      max_min_data[idx_in_block] = fmaxf(max_min_data[idx_in_block], max_min_data[idx_in_block + s]);
      min_data[idx_in_block + blockDim.x] = fminf(max_min_data[idx_in_block + blockDim.x], max_min_data[idx_in_block + blockDim.x + s]);
    }
    __syncthreads();
  }

  if (s == 0) {
    d_max_out[blockIdx.x] = max_data[0];
    d_min_out[blockIdx.x] = min_data[blockDim.x];
  }
}

__global__
void reduce(float * d_max_out,
            float * d_min_out
            float * d_max_intermediate,
            float * d_min_intermediate,
            float * d_in,
            int size) {
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    // if size is not divisible by maxThreadPerBlock, do I need an extra block
    int blocks = size / maxThreadsPerBlock;
    shmem_reduce_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>
            (d_max_intermediate, d_min_intermediate, d_in);
    threads = blocks;
    blocks = 1;
    shmem_reduce_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>
            (d_max_out, d_min_out, d_intermediate);
}

__global__
void find_max_and_min(float * d_in, float * global_max, float * global_min,
                      size_t numRows, size_t numCols) {
  const int ARRAY_SIZE = numRows * numCols;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
  float *d_max_intermediate, *d_min_intermediate;

  checkCudaErrors(cudaMalloc((void **) &d_max_intermediate, ARRAY_BYTES));
  checkCudaErrors(cudaMalloc((void **) &d_min_intermediate, ARRAY_BYTES));

  reduce(global_max, global_min, d_max_intermediate, d_min_intermediate, d_in, ARRAY_SIZE);

  checkCudaErrors(cudaFree(d_max_intermediate));
  checkCudaErrors(cudaFree(d_min_intermediate));
}

void scatter_kernel(const float * d_in, const size_t numBins, const float lumRange, int *d_bins) {

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  find_max_and_min(d_logLuminance, &max_logLum, &min_logLum, numRows, numCols);

  float lumRange = max_logLum - min_logLum;

}
