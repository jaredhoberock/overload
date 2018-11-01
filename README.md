A standalone CUDA-compatible C++11 implementation of overload.

# Demo

In C++11:

```c++
#include "overload.hpp"
#include <string>
#include <iostream>

int main()
{
  auto callme_with_int_or_string = overload(
    [](int value)
    {
      std::cout << "Received an integer: " << value << std::endl;
    },
    [](std::string value)
    {
      std::cout << "Received a string: " << value << std::endl;
    }
  );

  callme_with_int_or_string(42);
  callme_with_int_or_string("Hello, world!");

  std::cout << "OK" << std::endl;
}
```

Program output:
```
$ clang -std=c++11 demo.cpp -lstdc++
$ ./a.out 
Received an integer: 42
Received a string: Hello, world!
OK

```

And CUDA C++11:

```c++
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
```

Program output:
```
$ nvcc -std=c++11 --expt-extended-lambda demo.cu
$ ./a.out 
Received an integer: 42
Received a float: 1000.000000
OK
```

