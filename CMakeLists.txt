cmake_minimum_required(VERSION 3.13)
project(armnn_bench)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

message(STATUS ${ARMNN_INSTALL_DIR})
if (ARMNN_INSTALL_DIR)
    find_library(ARMNN_LIB libarmnn.so ${ARMNN_INSTALL_DIR})
    find_library(ARMNN_DELEGATE_LIB libarmnnDelegate.so ${ARMNN_INSTALL_DIR})
    find_library(ARMNN_ONNXPARSER_LIB libarmnnOnnxParser.so ${ARMNN_INSTALL_DIR})
    find_library(ARMNN_LIBARMNNTFLITEPARSER_LIB libarmnnTfLiteParser.so ${ARMNN_INSTALL_DIR})
    message(STATUS ${ARMNN_LIB})
    set(ARMNN_INCLUDE_DIR ${ARMNN_INSTALL_DIR}/include)
    message(STATUS ${ARMNN_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Could not find ARM-NN install directory")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# file(GLOB sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
# file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${ARMNN_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE ${ARMNN_LIB} ${ARMNN_DELEGATE_LIB} ${ARMNN_ONNXPARSER_LIB} ${ARMNN_LIBARMNNTFLITEPARSER_LIB} -lpthread)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 LINKER_LANGUAGE CXX)
