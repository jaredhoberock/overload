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

