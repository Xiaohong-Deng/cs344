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
void shmem_reduce_kernel(float * d_max_min_out, const float * d_in, int size) {
  
  // assert(blockDim.x % 2 == 0);
  // according to stackoverflow, extern __shared__ array can only have one copy
  // in order to fulfill the need of 2 array, double the size
  extern __shared__ float max_min_data[];
  
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_in_block = threadIdx.x;
  
  if (index >= size) return;

  max_min_data[idx_in_block] = d_in[index];
  max_min_data[idx_in_block + blockDim.x] = d_in[index];
  __syncthreads();
  
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (idx_in_block < s) {
      max_min_data[idx_in_block] = fmaxf(max_min_data[idx_in_block], max_min_data[idx_in_block + s]);
      max_min_data[idx_in_block + blockDim.x] = fminf(max_min_data[idx_in_block + blockDim.x], max_min_data[idx_in_block + blockDim.x + s]);
    }
    __syncthreads();
  }

  if (idx_in_block == 0) {
    d_max_min_out[blockIdx.x] = max_min_data[0];
    d_max_min_out[blockIdx.x + gridDim.x] = max_min_data[blockDim.x];
  }
}

__global__
void shmem_global_reduce_kernel(float *d_max_min_out, const float *d_in) {
  extern __shared__ float max_min_data[];
  
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_in_block = threadIdx.x;
  
  max_min_data[idx_in_block] = d_in[index];
  max_min_data[idx_in_block + blockDim.x] = d_in[index + blockDim.x];
  __syncthreads();
  
  int isOdd = 0;

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (idx_in_block < s) {
      max_min_data[idx_in_block] = fmaxf(max_min_data[idx_in_block], max_min_data[idx_in_block + s]);
      max_min_data[idx_in_block + blockDim.x] = fminf(max_min_data[idx_in_block + blockDim.x], max_min_data[idx_in_block + blockDim.x + s]);
      if (isOdd == 1 && idx_in_block == 0) {
        isOdd = 0;
        max_min_data[idx_in_block] = fmaxf(max_min_data[idx_in_block], max_min_data[idx_in_block + s + 1]);
        max_min_data[idx_in_block + blockDim.x] = fminf(max_min_data[idx_in_block + blockDim.x], max_min_data[idx_in_block + blockDim.x + s + 1]);
      }
    }
    if (s != 1 && s % 2 != 0) {
      isOdd = 1;
    }
    __syncthreads();
  }

  if (idx_in_block == 0) {
    d_max_min_out[blockIdx.x] = max_min_data[0];
    d_max_min_out[blockIdx.x + gridDim.x] = max_min_data[blockDim.x];
  }
}

void reduce(float * d_max_out,
            float * d_min_out,
            const float * d_in,
            int size) {
  float * d_max_min;
  checkCudaErrors(cudaMalloc((void **) &d_max_min, 2 * sizeof(float)));
  
  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  // if size is not divisible by maxThreadPerBlock, do I need an extra block
  int blocks = size / maxThreadsPerBlock;
  assert(blocks != 0);
  assert(blocks % 2 == 0);

  float *d_max_min_intermediate;
  // size of intermediate array should be number of blocks
  checkCudaErrors(cudaMalloc((void **) &d_max_min_intermediate, 2 * blocks * sizeof(float)));

  shmem_reduce_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>
          (d_max_min_intermediate, d_in, size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
  threads = blocks;
  assert(threads <= 1024);
  assert(threads <= 1024 * 6);
  blocks = 1;
  shmem_global_reduce_kernel<<<blocks, threads, 2 * threads * sizeof(float)>>>
          (d_max_min, d_max_min_intermediate);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  *d_max_out = d_max_min[0];
  *d_min_out = d_max_min[1];

  checkCudaErrors(cudaFree(d_max_min_intermediate));
  checkCudaErrors(cudaFree(d_max_min));
}

void find_max_and_min(const float * d_in, float * global_max, float * global_min,
                      size_t numRows, size_t numCols) {
  const int ARRAY_SIZE = numRows * numCols;
  //const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


  reduce(global_max, global_min, d_in, ARRAY_SIZE);
}

__global__
void scatter_kernel(const float *d_in, const size_t numBins, size,
                    const float lumRange, unsigned int *d_bins, const float lumMin) {
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if (myId >= size) return;
  float myItem = d_in[myId];
  unsigned int myBin = min((unsigned int) (numBins - 1), (unsigned int) ((myItem - lumMin) / lumRange * numBins));
  atomicAdd(&(d_bins[myBin]), 1);
}

__global__
void init_bins(unsigned int *d_bins) {
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  d_bins[myId] = 0;
}

void create_histos(unsigned int *d_bins, size_t numBins, float lumRange, float lumMin, const float *d_in,
                    size_t numRows, size_t numCols) {

  init_bins<<<1, numBins>>>(d_bins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  const int maxThreadsPerBlock = 1024;
  int threads = maxThreadsPerBlock;
  // if size is not divisible by maxThreadPerBlock, do I need an extra block
  int size = numRows * numCols;
  int blocks = size / maxThreadsPerBlock;  
  scatter_kernel<<<blocks, threads>>>(d_in, numBins, size, lumRange, d_bins, lumMin);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

__global__
void exclusive_prefix_scan_kernel(const unsigned int* const d_in, unsigned int *d_out,
                                  size_t threadsPerBlock) {
  extern __shared__ float sdata[];

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int myId = threadIdx.x;

  sdata[myId] = d_in[index];
  __syncthreads();
  
  // upsweep
  for (int s = 1; ( 1 << s ) <= threadsPerBlock; ++s) {
    int stride = ( 1 << (s - 1) );
    int base = 1 << s;
    if ((myId + 1) % base == 0) {
      sdata[myId] = sdata[myId] + sdata[myId - stride];
    }
    __syncthreads();
  }

  // downsweep
  if (myId == threadsPerBlock - 1) {
    sdata[myId] = 0;
  }
  __syncthreads();

  for (int s = threadsPerBlock / 2; s > 0; s >>= 1) {
    int base = s << 1;
    if ((myId + 1) % base == 0) {
      int left = sdata[myId];
      int right = sdata[myId - s] + left;
      sdata[myId - s] = left;
      sdata[myId] = right; 
    }
    __syncthreads();
  }

  d_out[myId] = sdata[myId];
  __syncthreads();

}

void ex_pre_scan(unsigned int* d_cdf, const unsigned int *d_in, size_t numBins) {
  // how do I assert that numBins == power of 2
  assert(numBins <= 1024);
  assert(numBins * sizeof(unsigned int) <= 4096 * 12);
  exclusive_prefix_scan_kernel<<<1, numBins, numBins * sizeof(unsigned int)>>>(d_in, d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
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

  unsigned int *d_bins;

  checkCudaErrors(cudaMalloc((void **) &d_bins, numBins * sizeof(unsigned int)));

  create_histos(d_bins, numBins, lumRange, min_logLum, d_logLuminance, numRows, numCols);

  ex_pre_scan(d_cdf, d_bins, numBins);

  checkCudaErrors(cudaFree(d_bins));
}
