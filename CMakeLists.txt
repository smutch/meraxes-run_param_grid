cmake_minimum_required(VERSION 3.5)
project(run_param_grid)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -Wall -O0")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type.")

find_package(Meraxes REQUIRED)

set(SRC
    run_param_grid.c
    )

add_executable(run_param_grid ${SRC})
target_link_libraries(run_param_grid meraxes_lib)
set_target_properties(run_param_grid PROPERTIES
    C_STANDARD 99
    INSTALL_RPATH_USE_LINK_PATH TRUE
    )

install(TARGETS run_param_grid DESTINATION .)
