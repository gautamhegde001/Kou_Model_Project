cmake_minimum_required(VERSION 3.15)
project(kou_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_kou_mc
    src/cpp/kou_price_monte_carlo/kou_price_mc.cpp
    src/cpp/kou_price_monte_carlo/bindings.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_kou_mc PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _kou_mc LIBRARY DESTINATION pkgs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)