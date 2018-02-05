cmake_minimum_required(VERSION 3.5)
project(run_param_grid)

find_package(Meraxes REQUIRED)

set(SRC
    run_param_grid.c
    )

add_executable(run_param_grid ${SRC})
target_link_libraries(run_param_grid meraxes_lib)
set_property(TARGET run_param_grid PROPERTY C_STANDARD 99)

install(TARGETS run_param_grid DESTINATION .)
