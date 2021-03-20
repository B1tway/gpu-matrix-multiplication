#include <iostream>
#include <CL/cl.hpp>
#include <utility>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#define __CL_ENABLE_EXCEPTIONS
#define M 100
#define N 100
#define K 100
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
	const size_t gflops_kf = ((size_t) N * M * K * 2) / (1000 * 1000 * 1000);
	float* a = new float[K * M];
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < K; j++)
		{
			a[i * K + j] = rand() / 1e5;
			//a[i * COLUMNS + j] = 1;

		}
	}
	float* b = new float[K * N];
	for (size_t i = 0; i < K; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			b[i * N + j] = rand() / 1e5;
			//b[i * COLUMNS + j] = 1;

		}
	}
	float* res = new float[M * N];
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			res[i * N + j] = 0;
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
	std::string kernel_source = read_kernel("kernel1.cl");
	cl::Program program(context, kernel_source, true);
	if (program.build({ current_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(current_device) << "\n";
		getchar();
		exit(1);
	}
	int32_t gr = M;
	if (M % TILE_SIZE > 0)
		gr = M + (TILE_SIZE - M % TILE_SIZE);
	int32_t gc = N;
	if (N % TILE_SIZE > 0)
		gc = N + (TILE_SIZE - N % TILE_SIZE);
	cl::Event event;
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float) * M * K);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(float) * K * N);
	cl::Buffer buffer_RES(context, CL_MEM_READ_WRITE, sizeof(float) * N * M);
	cl::CommandQueue queue(context, current_device);
	errcode = queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * M * K, &a[0]);
	OCL_SAFE_CALL(errcode);
	errcode = queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * K * N, &b[0]);
	OCL_SAFE_CALL(errcode);
	cl::Kernel kernel = cl::Kernel(program, "simple_mul");
	errcode = kernel.setArg(0, buffer_A);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(1, buffer_B);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(2, buffer_RES);
	OCL_SAFE_CALL(errcode);
	int32_t N_value = N;
	int32_t M_value = M;
	int32_t K_value = K;
	errcode = kernel.setArg(3, sizeof(int), &M_value);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(4, sizeof(int), &N_value);
	OCL_SAFE_CALL(errcode);
	errcode = kernel.setArg(5, sizeof(int), &K_value);
	OCL_SAFE_CALL(errcode);
	cl::NDRange global(gr, gc);
	cl::NDRange local(TILE_SIZE, TILE_SIZE);
	using milli = std::chrono::milliseconds;
	auto start = std::chrono::high_resolution_clock::now();
	errcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
	OCL_SAFE_CALL(errcode);
	errcode = queue.finish();
	OCL_SAFE_CALL(errcode);
	cl_ulong startTime;
	cl_ulong endTime;
	event.getProfilingInfo(CL_PROFILING_COMMAND_START, &startTime);
	event.getProfilingInfo(CL_PROFILING_COMMAND_END, &endTime);
	float milliseconds = (endTime - startTime) / 1000000.0;
	auto finish = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
	//float time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
	float time_sec = time / 1e6;
	std::cout << "Execution took " << time / 1e6 << " milliseconds\n";
	std::cout << "Gflops " << gflops_kf / (time_sec) * 1000 << "\n";

	errcode = queue.enqueueReadBuffer(buffer_RES, CL_TRUE, 0, sizeof(float) * M * N, &res[0]);
	OCL_SAFE_CALL(errcode);

	std::cout << std::setprecision(3);

	//std::cout << "\nMatrix #1: \n";
	//for (size_t i = 0; i < ROWS; i++)  {
	//	for (size_t j = 0; j < COLUMNS; j++) {
	//		std::cout << a[i * COLUMNS + j] << " ";
	//	}
	//	std::cout << "\n";
	//}
	//std::cout << "\nMatrix #2: \n";
	//for (size_t i = 0; i < ROWS; i++)
	//{
	//	for (size_t j = 0; j < COLUMNS; j++)
	//	{
	//		std::cout << b[i * COLUMNS + j] << " ";
	//	}
	//	std::cout << "\n";
	//}


	/*std::cout << "\nMatrix #RESULT: \n";
	for (size_t i = 0; i < ROWS; i++)
	{
		for (size_t j = 0; j < COLUMNS; j++)
		{
			std::cout << res[i * COLUMNS + j] << " ";
		}
		std::cout << "\n";
	}
	float check = 0.0f;
	for (size_t i = 0; i < ROWS; i++) {
		check += a[i] * b[i*COLUMNS];
	}
	std::cout << std::endl;
	std::cout << check << std::endl;*/
	free(a);
	free(b);
	free(res);

}

int main() {
	run_kernel("kernel2_1.cl", "local_mul");
}


