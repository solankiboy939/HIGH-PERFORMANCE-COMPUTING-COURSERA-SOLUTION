/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "memory_copy.h"

__global__ void div(float *d_a, float *d_b, float *h_c, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        h_c[i] = d_a[i] / d_b[i];
    }
}


__host__ std::tuple<float *, float *> allocateRandomHostMemory(int numElements)
{
    srand(time(0));
    size_t size = numElements * sizeof(float);

    float *h_a;
    cudaHostAlloc((void **)&h_a, size, cudaHostAllocMapped);

    float *h_b;
    cudaHostAlloc((void **)&h_b, size, cudaHostAllocMapped);

    for (int i = 0; i < numElements; ++i)
    {
        h_a[i] = LO_RAND + static_cast<float>(rand())/(static_cast<float>(RAND_MAX/(HI_RAND-LO_RAND)));
        h_b[i] = LO_RAND + static_cast<float>(rand())/(static_cast<float>(RAND_MAX/(HI_RAND-LO_RAND)));
    }

    return {h_a, h_b};
}


// Based heavily on [https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/](https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/)
__host__ std::tuple<float *, float *, int> readCsv(std::string filename)
{
    vector<int> tempResult;
    ifstream myFile(filename);
    if(!myFile.is_open()) throw runtime_error("Could not open file");
    string line;
    float val;

    getline(myFile, line);
    stringstream ss0(line);
    while (ss0 >> val) {
        tempResult.push_back(val);
        if (ss0.peek() == ',') ss0.ignore();
    }
    int numElements = tempResult.size();
    size_t size = numElements * sizeof(float);

    float *h_a;
    cudaHostAlloc((void **)&h_a, size, cudaHostAllocMapped);
    copy(tempResult.begin(), tempResult.end(), h_a);
    tempResult.clear();

    getline(myFile, line);
    stringstream ss1(line);
    while (ss1 >> val) {
        tempResult.push_back(val);
        if (ss1.peek() == ',') ss1.ignore();
    }
    float *h_b;
    cudaHostAlloc((void **)&h_b, size, cudaHostAllocMapped);
    copy(tempResult.begin(), tempResult.end(), h_b);

    myFile.close();
    return {h_a, h_b, numElements};
}


__host__ std::tuple<float *, float *> allocateDeviceMemory(int numElements)
{
    // Allocate device pointers as mapped pointers to host memory
    float *d_a = NULL;
    float *d_b = NULL;

    // Map host memory to device pointers using cudaHostGetDevicePointer
    size_t size = numElements * sizeof(float);

    // You will set d_a and d_b in the host->device copy function instead
    return {d_a, d_b};
}

__host__ void copyMappedFromHostToDevice(float *h_a, float *h_b, float *&d_a, float *&d_b, int numElements)
{
    cudaHostGetDevicePointer((void**)&d_a, h_a, 0);
    cudaHostGetDevicePointer((void**)&d_b, h_b, 0);
}



__host__ void executeKernel(float *d_a, float *d_b, float *h_c, int numElements, int threadsPerBlock)
{
    // Launch the CUDA Kernel (no need for additional copy)
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    div<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, h_c, numElements);



    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch div kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory (use cudaFreeHost for mapped memory)
__host__ void deallocateMemory(float *d_a, float *d_b)
{
    // Actually, you should free host-mapped memory using cudaFreeHost, not cudaFree
    cudaError_t err = cudaFreeHost(d_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free host mapped vector h_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFreeHost(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free host mapped vector h_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// The rest of your output function and main routine remain unchanged!


// Note that this function should not be modified
__host__ void outputToFile(std::string currentPartId, float *h_a, float *h_b, float *h_c, int numElements)
{
	string outputFileName = "output-" + currentPartId + ".txt";
	// NOTE: Do not remove this output to file statement as it is used to grade assignment,
	// so it should be called by each thread
	ofstream outputFile;
	outputFile.open (outputFileName, ofstream::app);

    outputFile << "PartID: " << currentPartId << "\n";
    outputFile << "Input A: ";
    for (int i = 0; i < numElements; ++i)
        outputFile << h_a[i] << " ";
    outputFile << "\n";
    outputFile << "Input B: ";
    for (int i = 0; i < numElements; ++i)
        outputFile << h_b[i] << " ";
    outputFile << "\n";
    outputFile << "Result: ";
    for (int i = 0; i < numElements; ++i)
        outputFile << h_c[i] << " ";
    outputFile << "\n";

	outputFile.close();
}

// Note that this function should not be modified
__host__ std::tuple<int, std::string, int, std::string, std::string> parseCommandLineArguments(int argc, char *argv[])
{
    int numElements = 10;
    int threadsPerBlock = 256;
    std::string currentPartId = "test";
    std::string mathematicalOperation = "add";
    std::string inputFilename = "NULL";

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-t") == 0) 
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if(option.compare("-n") == 0) 
        {
            numElements = atoi(value.c_str());
        }
        else if(option.compare("-f") == 0) 
        {
            inputFilename = value;
        }
        else if(option.compare("-p") == 0) 
        {
            currentPartId = value;
        }
        else if(option.compare("-o") == 0) 
        {
            mathematicalOperation = value;
        }
    }

    return {numElements, currentPartId, threadsPerBlock, inputFilename, mathematicalOperation};
}

__host__ std::tuple<float *, float *, int> setUpInput(std::string inputFilename, int numElements)
{
    srand(time(0));
    float *h_a;
    float *h_b;

    if(inputFilename.compare("NULL") != 0)
    {
        tuple<float *, float*, int>csvData = readCsv(inputFilename);
        h_a = get<0>(csvData);
        h_b = get<1>(csvData);
        numElements = get<2>(csvData);
    }
    else 
    {
        tuple<float *, float*> randomData = allocateRandomHostMemory(numElements);
        h_a = get<0>(randomData);
        h_b = get<1>(randomData);
    }

    return {h_a, h_b, numElements};
}

/*
 * Host main routine
 * -n numElements - the number of elements of random data to create
 * -f inputFile - the file for non-random input data
 * -o mathematicalOperation - this will decide which math operation kernel will be executed
 * -p currentPartId - the Coursera Part ID
 * -t threadsPerBlock - the number of threads to schedule for concurrent processing
 * Note that this function should not be modified
 */
int main(int argc, char *argv[])
{
    auto[numElements, currentPartId, threadsPerBlock, inputFilename, mathematicalOperation] = parseCommandLineArguments(argc, argv);
    tuple<float *, float*, int> searchInputTuple = setUpInput(inputFilename, numElements);
    float *h_a;
    float *h_b;

    h_a = get<0>(searchInputTuple);
    h_b = get<1>(searchInputTuple);
    numElements = get<2>(searchInputTuple);

    float *h_c;
    cudaMallocManaged((float **)&h_c, numElements*sizeof(float));

    float *d_a = nullptr, *d_b = nullptr;
    copyMappedFromHostToDevice(h_a, h_b, d_a, d_b, numElements);

    executeKernel(d_a, d_b, h_c, numElements, threadsPerBlock);

    outputToFile(currentPartId, h_a, h_b, h_c, numElements);

    deallocateMemory(d_a, d_b);

    cleanUpDevice();
    return 0;
}
