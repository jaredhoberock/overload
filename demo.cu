#include "overload.hpp"
#include <iostream>

template<class F, class T>
__global__ void kernel(F f, T value)
{
  f(value);
}


int main()
{
  auto callme_with_int_or_float = overload(
    [] __host__ __device__ (int value)
    {
      printf("Received an integer: %d\n", value);
    },
    [] __host__ __device__ (float value)
    {
      printf("Received a float: %f\n", value);
    }
  );

  kernel<<<1,1>>>(callme_with_int_or_float, 42);
  kernel<<<1,1>>>(callme_with_int_or_float, 1000.f);

  cudaDeviceSynchronize();

  std::cout << "OK" << std::endl;
}

