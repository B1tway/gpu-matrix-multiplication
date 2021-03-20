#include <iostream>
#include <CL/cl.hpp>
#include <utility>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#define __CL_ENABLE_EXCEPTIONS
#define ROWS (1000)    // ROWS of vectors a, b, and c
#define COLUMNS (1500)
#define TILE_SIZE (32)
void reportError(cl_int err, const std::string& filename, int line)
{
	if (CL_SUCCESS == err)
		return;
	std::string message = "OpenCL error code " + std::to_string(err) + " encountered at " + filename + ":" + std::to_string(line);
	throw std::runtime_error(message);
}
#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


std::string read_kernel(std::string filename) {
	std::ifstream input_file(filename);
	if (!input_file.is_open()) {
		std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
		exit(EXIT_FAILURE);
	}
	return std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}
void run_kernel(std::string filename, std::string kernel_name) {
	cl_int errcode;
	const size_t gflops_kf = ((size_t)ROWS * COLUMNS * ROWS * 2) / (1000 * 1000 * 1000);
	float* a = new float[ROWS * COLUMNS];
	int N = ROWS;
	int M = COLUMNS;
	for (size_t i = 0; i < ROWS; i++)
	{
		for (size_t j = 0; j < COLUMNS; j++)
		{
			a[i * COLUMNS + j] = rand() / 1e5;
			//a[i * COLUMNS + j] = 1;

		}
	}
	float* res = new float[ROWS * COLUMNS];
	for (size_t i = 0; i < ROWS; i++)
	{
		for (size_t j = 0; j < COLUMNS; j++)
		{
			res[i * COLUMNS + j] = 0;
		}
	}
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << "No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform current_platform = platforms[1];
	std::cout << "Using platform: " << current_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
	std::vector<cl::Device> all_devices;
	errcode = current_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	OCL_SAFE_CALL(errcode);
	if (all_devices.size() == 0) { // Check for issues
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device current_device = all_devices[0];
	std::cout << "Using device: " << current_device.getInfo<CL_DEVICE_NAME>() << "\n";
	std::cout << current_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
	cl::Context context({ current_device });
	std::string kernel_source = read_kernel(filename);
	cl::Program program(context, kernel_source, true);
	if (program.build({ current_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(current_device) << "\n";
		getchar();
		exit(1);
	}
	int32_t gr = ROWS;
	if (ROWS % TILE_SIZE > 0)
		gr = ROWS + (TILE_SIZE - ROWS % TILE_SIZE);
	int32_t gc = COLUMNS;
	if (COLUMNS % TILE_SIZE > 0)
		gc = COLUMNS + (TILE_SIZE - COLUMNS % TILE_SIZE);
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float) * ROWS * COLUMNS);
	cl::Buffer buffer_RES(context, CL_MEM_READ_WRITE, sizeof(float) * ROWS * COLUMNS);
	cl::CommandQueue queue(context, current_device);
	errcode = queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &a[0]);
	OCL_SAFE_CALL(errcode);
	errcode = queue.enqueueWriteBuffer(buffer_RES, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &res[0]);
	OCL_SAFE_CALL(errcode);
	 
	cl::Kernel kernel = cl::Kernel(program, kernel_name.c_str());
	errcode = kernel.setArg(0, buffer_A);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(1, buffer_RES);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(2, sizeof(int), &N);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(3, sizeof(int), &M);
	OCL_SAFE_CALL(errcode);
	std::cout << "init kernel " << kernel_name << std::endl;
	std::cout << gr << " " << gc << std::endl;
	cl::NDRange global(gr, gc);
	cl::NDRange local(TILE_SIZE, TILE_SIZE);
	using milli = std::chrono::milliseconds;
	auto start = std::chrono::high_resolution_clock::now();
	cl::Event event;
	errcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	OCL_SAFE_CALL(errcode);
	errcode = queue.finish();
	OCL_SAFE_CALL(errcode);
	auto finish = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	//float time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
	float time_sec = time / 1e6;
	std::cout << "Execution took " << time / 1e6 << " milliseconds\n";
	std::cout << "Gflops " << gflops_kf / (time_sec) * 1000 << "\n";

	errcode = queue.enqueueReadBuffer(buffer_RES, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &res[0]);
	OCL_SAFE_CALL(errcode);
	std::cout << std::setprecision(3);

	//std::cout << "\nMatrix #1: \n";
	//for (size_t i = 0; i < ROWS; i++)  {
	//	for (size_t j = 0; j < COLUMNS; j++) {
	//		std::cout << a[i * COLUMNS + j] << " ";
	//	}
	//	std::cout << "\n";
	//}


	//std::cout << "\nMatrix #RESULT: \n";
	//for (size_t i = 0; i < ROWS; i++)
	//{
	//	for (size_t j = 0; j < COLUMNS; j++)
	//	{
	//		std::cout << res[i * COLUMNS + j] << " ";
	//	}
	//	std::cout << "\n";
	//}
	free(a);
	free(res);
}

int main() {
	run_kernel("copy.cl", "matrix_copy");
}


