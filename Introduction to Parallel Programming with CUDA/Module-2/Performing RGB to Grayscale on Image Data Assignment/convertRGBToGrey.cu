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
#include "convertRGBToGrey.hpp"
#include <fstream>  // <-- ADD THIS FOR FILE OUTPUT

/*
 * CUDA Kernel Device code
 *
 */
__global__ void convert(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = d_rows * d_columns;

    if (idx < totalPixels) {
        d_gray[idx] = (d_r[idx] + d_g[idx] + d_b[idx]) / 3;
    }
}

__host__ float compareGrayImages(uchar *gray, uchar *test_gray, int rows, int columns)
{
    cout << "Comparing actual and test grayscale pixel arrays\n";
    int numImagePixels = rows * columns;
    int imagePixelDifference = 0;

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            uchar image0Pixel = gray[r*columns + c];
            uchar image1Pixel = test_gray[r*columns + c];
            imagePixelDifference += abs(image0Pixel - image1Pixel);
        }
    }

    float meanImagePixelDifference = static_cast<float>(imagePixelDifference) / numImagePixels;
    float scaledMeanDifferencePercentage = meanImagePixelDifference / 255.0f;
    printf("meanImagePixelDifference: %f scaledMeanDifferencePercentage: %f\n", meanImagePixelDifference, scaledMeanDifferencePercentage);
    return scaledMeanDifferencePercentage;
}

__host__ std::tuple<uchar *, uchar *, uchar *, uchar *> allocateDeviceMemory(int rows, int columns)
{
    cout << "Allocating GPU device memory\n";
    int num_image_pixels = rows * columns;
    size_t size = num_image_pixels * sizeof(uchar);

    uchar *d_r = NULL;
    cudaError_t err = cudaMalloc(&d_r, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    uchar *d_g = NULL;
    err = cudaMalloc(&d_g, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    uchar *d_b = NULL;
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    uchar *d_gray = NULL;
    err = cudaMalloc(&d_gray, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_gray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);

    return {d_r, d_g, d_b, d_gray};
}

__host__ void copyFromHostToDevice(uchar *h_r, uchar *d_r, uchar *h_g, uchar *d_g, uchar *h_b, uchar *d_b, int rows, int columns)
{
    cout << "Copying from Host to Device\n";
    int num_image_pixels = rows * columns;
    size_t size = num_image_pixels * sizeof(uchar);

    cudaError_t err;
    err = cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector r from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector b from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray, int rows, int columns, int threadsPerBlock)
{
    cout << "Executing kernel\n";
    int totalPixels = rows * columns;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    convert<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_g, d_b, d_gray);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
}

__host__ void copyFromDeviceToHost(uchar *d_gray, uchar *gray, int rows, int columns)
{
    cout << "Copying from Device to Host\n";
    size_t size = rows * columns * sizeof(uchar);

    cudaError_t err = cudaMemcpy(gray, d_gray, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array d_gray from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void deallocateMemory(uchar *d_r, uchar *d_g, uchar *d_b, uchar *d_gray)
{
    cout << "Deallocating GPU device memory\n";
    cudaError_t err = cudaFree(d_r);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_gray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_gray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    std::string inputImage = "sloth.png";
    std::string outputImage = "grey-sloth.png";
    std::string currentPartId = "test";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " currentPartId: " << currentPartId << " threadsPerBlock: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, currentPartId, threadsPerBlock};
}

__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);
    
    const int rows = img.rows;
    const int columns = img.cols;

    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    uchar *h_r = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *h_g = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *h_b = (uchar *)malloc(sizeof(uchar) * rows * columns);

    if (!h_r || !h_g || !h_b) {
        fprintf(stderr, "Failed to allocate host memory for image channels.\n");
        exit(EXIT_FAILURE);
    }
    
    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            Vec3b intensity = img.at<Vec3b>(r, c);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            h_r[r*columns + c] = red;
            h_g[r*columns + c] = green;
            h_b[r*columns + c] = blue;
        }
    }

    return {rows, columns, h_r, h_g, h_b};
}

__host__ uchar *cpuConvertToGray(std::string inputFile)
{
    cout << "CPU converting image file to grayscale\n";
    Mat grayImage = imread(inputFile, IMREAD_GRAYSCALE);
    const int rows = grayImage.rows;
    const int columns = grayImage.cols;

    uchar *gray = (uchar *)malloc(sizeof(uchar) * rows * columns);

    if (!gray) {
        fprintf(stderr, "Failed to allocate host memory for CPU grayscale image.\n");
        exit(EXIT_FAILURE);
    }

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            gray[r*columns + c] = min(grayImage.at<uchar>(r, c), 254);
        }
    }

    return gray;
}

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    std::string currentPartId = get<2>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<3>(parsedCommandLineArgsTuple);

    // Declare this OUTSIDE try so we can use it later
    float scaledMeanDifferencePercentage = 0.0f;

    try 
    {
        auto[rows, columns, h_r, h_g, h_b] = readImageFromFile(inputImage);
        uchar *gray = (uchar *)malloc(sizeof(uchar) * rows * columns);
        if (!gray) {
            fprintf(stderr, "Failed to allocate host memory for output grayscale image.\n");
            exit(EXIT_FAILURE);
        }

        std::tuple<uchar *, uchar *, uchar *, uchar *> memoryTuple = allocateDeviceMemory(rows, columns);
        uchar *d_r = get<0>(memoryTuple);
        uchar *d_g = get<1>(memoryTuple);
        uchar *d_b = get<2>(memoryTuple);
        uchar *d_gray = get<3>(memoryTuple);

        copyFromHostToDevice(h_r, d_r, h_g, d_g, h_b, d_b, rows, columns);
        executeKernel(d_r, d_g, d_b, d_gray, rows, columns, threadsPerBlock);
        copyFromDeviceToHost(d_gray, gray, rows, columns);
        deallocateMemory(d_r, d_g, d_b, d_gray);
        cleanUpDevice();

        Mat grayImageMat(rows, columns, CV_8UC1);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for(int r = 0; r < rows; ++r)
        {
            for(int c = 0; c < columns; ++c)
            {
                grayImageMat.at<uchar>(r,c) = gray[r*columns + c];
            }
        }

        imwrite(outputImage, grayImageMat, compression_params);

        uchar *test_gray = cpuConvertToGray(inputImage);
        scaledMeanDifferencePercentage = compareGrayImages(gray, test_gray, rows, columns) * 100;
        cout << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";

        free(h_r);
        free(h_g);
        free(h_b);
        free(gray);
        free(test_gray);
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }

    // ✅ NOW THIS BLOCK WORKS — inside main, after try, and scaledMeanDifferencePercentage is in scope
    {
        std::string output_filename = "output-test.txt";
        if (!currentPartId.empty() && currentPartId != "test") {
            output_filename = "output-" + currentPartId + ".txt";
        }
        std::ofstream out_file(output_filename);
        if (out_file.is_open()) {
            out_file << "inputImage: " << inputImage 
                     << " outputImage: " << outputImage 
                     << " currentPartId: " << currentPartId 
                     << " threadsPerBlock: " << threadsPerBlock << "\n";
            out_file << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";
            out_file.close();
            std::cout << "Wrote output to " << output_filename << std::endl;
        }
    }

    return 0;
}