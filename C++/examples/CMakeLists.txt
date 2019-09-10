# CPL-SLAM command-line driver
add_executable(cpl_slam cpl_slam.cpp)
target_link_libraries(cpl_slam SESync CPL-SLAM)

if(${ENABLE_PROFILING})
target_compile_definitions(CPL-SLAM PRIVATE GPERFTOOLS)
target_link_libraries(CPL-SLAM profiler)
endif()

if(${ENABLE_OPENMP})
if(OPENMP_FOUND)
set_target_properties(CPL-SLAM PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
endif()
endif()

message(STATUS "Building main CPL-SLAM command-line executable in directory ${EXECUTABLE_OUTPUT_PATH}\n")
