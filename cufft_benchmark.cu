#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <time.h>
#include <math.h>
#include <nvml.h>

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA error %d: %s\n", __FILE__, __LINE__, result, cudaGetErrorString(result)); \
        exit(1); \
    } \
} while (0)

#define CHECK_CUFFT_ERROR(call) \
do { \
    cufftResult result = call; \
    if (result != CUFFT_SUCCESS) { \
        fprintf(stderr, "%s:%d: cuFFT error %d\n", __FILE__, __LINE__, result); \
        exit(1); \
    } \
} while (0)

// Load lengths of different trials from a file, including descriptions
#define LINE_LEN 128
int load_lengths(const char* filename, int* ntrials, long long** nffts, char (**description)[LINE_LEN]) {
    int count = 0;
    char buffer[LINE_LEN];
    FILE* f = fopen(filename, "r");
    if (!f) {
       fprintf(stderr, "Failed to read text file %s containing lengths for different trials.\n", filename);
       perror(filename);
       return 1;
    }
    // First pass: count lines
    while (fgets(buffer, sizeof(buffer), f)) count++;
    *ntrials = count; 
    // Allocate memory
    *nffts = (long long*) malloc(count * sizeof(**nffts));
    *description = (char (*)[LINE_LEN]) malloc(count * sizeof(**description));
    if (!*nffts || !*description) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(f);
        return 2;
    }
    // Second pass: read and parse
    rewind(f);
    for (int i = 0; i < count; i++) {
        if (!fgets(buffer, sizeof(buffer), f)) break;
        sscanf(buffer, "%lld =", &((*nffts)[i]));
        strncpy((*description)[i], buffer, LINE_LEN);
        (*description)[i][LINE_LEN - 1] = '\0';
    }
    fclose(f);
    return 0;
}

// For standard C sorting
int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);  // returns 1, 0, or -1
}

int main(int argc, char **argv) {
    long long num_iterations = 100;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr,"CUDA version: %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);
    fprintf(stderr,"GPU: %s\n", prop.name);
    fprintf(stderr,"Driver compute compatibility: %d.%d\n", prop.major, prop.minor);
    // Initialize NVML library
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr,"Failed to initialize NVML library: %s\n", nvmlErrorString(result));
        return 1;
    }
    char version_str[NVML_DEVICE_PART_NUMBER_BUFFER_SIZE+1];
    nvmlReturn_t retval = nvmlSystemGetDriverVersion(version_str, NVML_DEVICE_PART_NUMBER_BUFFER_SIZE);
    if (retval != NVML_SUCCESS) {
        fprintf(stderr, "%s\n",nvmlErrorString(retval));
        return 1;
    }
    fprintf(stderr,"Driver version: %s\n", version_str);

    // Read file containing the different FFT lengths to try
    int ntrials;
    long long* nffts;
    char (*description)[LINE_LEN];
    if (load_lengths("Lengths.txt", &ntrials, &nffts, &description) != 0) {
       fprintf(stderr, "Failed to load Lengths.txt\n");
       exit(1);
    }
    printf("Testing %d FFT lengths, ranging from %lld to %lld.\n", ntrials, nffts[0], nffts[ntrials-1]);
    srand(time(NULL)); // initialize seed

    float* times = (float*) malloc(num_iterations * sizeof(*times));
    if (!times) {
        fprintf(stderr, "Failed to allocate timing buffer\n");
        return 1;
    }

    // Loop over the different FFT trial lengths
    for (int i = 0; i < ntrials; i++){
        long long n = nffts[i];
        fprintf(stderr,"**************************************\n");
        fprintf(stderr,"N-point FFT: %s\n", description[i]);
        fprintf(stderr,"Number of iterations: %lld\n", num_iterations);

        int batch = 1;
        int rank = 1;
        int istride = 1, ostride = 1;
        long long idist = n, odist = n;
        long long nembed[1] = {n}, inembed[1] = {n}, onembed[1] = {n};
        cufftHandle forward_plan;
        cudaEvent_t start, stop;
        float *input_data, *host_input_data;
        cufftComplex *fft_data;
        float mean_time, median_time;
        size_t work_size;

	float input_size_gb = n * sizeof(float)/1e9;
	float output_size_gb = (n/2 + 1)*sizeof(cufftComplex)/1e9;
        fprintf(stderr,"Input float array size: %lf GB\n", input_size_gb);
        fprintf(stderr,"Output complex array size: %lf GB\n", output_size_gb);

        // Allocate memory on host
	host_input_data = (float*) malloc(n * batch * sizeof(*host_input_data));
	if (!host_input_data) {
	   fprintf(stderr, "Failed to allocate input data array\n");
	   exit(1);
	}

        // Initialize input data on host
        for (long int k = 0; k < n * batch; k++) {
            host_input_data[k] = (float) rand() / RAND_MAX;
        }
        //get size estimate
	CHECK_CUFFT_ERROR(cufftEstimate1d(n, CUFFT_R2C, batch, &work_size));
        float work_size_gb = work_size/1.0e9;
        fprintf(stderr, "Work size estimate: %lf GB\n", work_size_gb);
        fprintf(stderr, "Total size estimate: %lf GB\n", input_size_gb + output_size_gb + work_size_gb);

        // Allocate memory on device
	CHECK_CUDA_ERROR(cudaMalloc((void**) &input_data, n * batch * sizeof(*input_data)));
	CHECK_CUDA_ERROR(cudaMalloc((void**) &fft_data, (n/2 + 1) * batch * sizeof(*fft_data)));

        // Create FFT plan
        CHECK_CUFFT_ERROR(cufftCreate(&forward_plan));
        CHECK_CUFFT_ERROR(cufftMakePlanMany64(forward_plan, rank, nembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch, &work_size));

        // Copy input data to device
        CHECK_CUDA_ERROR(cudaMemcpy(input_data, host_input_data, n * batch * sizeof(float), cudaMemcpyHostToDevice));

        // Record FFT times
        mean_time = 0.0;
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        for (int iter = 0; iter < num_iterations; iter++) {
            float elapsed_time = 0.0;
            CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

            CHECK_CUFFT_ERROR(cufftExecR2C(forward_plan, input_data, fft_data));

            CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
            mean_time += elapsed_time;    
            times[iter] = elapsed_time;    
        }
	CHECK_CUDA_ERROR(cudaEventDestroy(start));
	CHECK_CUDA_ERROR(cudaEventDestroy(stop));

	// compute median times and mean
	qsort(times, num_iterations, sizeof(float), compare_floats);
	if (num_iterations % 2 == 0) {
	   median_time = (times[num_iterations / 2 - 1] + times[num_iterations / 2]) / 2.0f;
	} else {
	   median_time = times[num_iterations / 2];
	}
        mean_time = mean_time / num_iterations;

        fprintf(stderr,"Mean time: %f ms for length %s\n", mean_time, description[i]);
        fprintf(stderr,"Median time: %f ms for length %s\n", median_time, description[i]);

        // Free memory
        free(host_input_data);
        CHECK_CUDA_ERROR(cudaFree(input_data));
        CHECK_CUDA_ERROR(cudaFree(fft_data));
        CHECK_CUFFT_ERROR(cufftDestroy(forward_plan));
    }
    return 0;
}
