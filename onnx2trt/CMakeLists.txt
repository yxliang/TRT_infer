cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(trtexec LANGUAGES CXX CUDA)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_DEBUG_POSTFIX "d")
SET(CMAKE_CUDA_ARCHITECTURES "52;61;75;86")
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON) 

# # For finding FindTensorRT.cmake
# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# SET(3rdlib_DIR "D:/Libraries/")
# SET(TensorRT_DIR "${3rdlib_DIR}/TensorRT")
# SET(OpenCV_DIR "${3rdlib_DIR}/OpenCV-cuda")
# MESSAGE("TensorRT_INCLUDE_DIR: ${TensorRT_INCLUDE_DIR}")
# # INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})

# #add ort
# add_library(libonnxruntime SHARED IMPORTED GLOBAL)
# set_target_properties(libonnxruntime PROPERTIES IMPORTED_IMPLIB "${3rdlib_DIR}/onnxruntime/lib/onnxruntime.lib" 
									    # IMPORTED_LOCATION "${3rdlib_DIR}/onnxruntime/lib/onnxruntime.dll"
										# INTERFACE_COMPILE_DEFINITIONS  "ORT_DLL_IMPORT;USE_ORT;USE_CUDA"
										# INTERFACE_INCLUDE_DIRECTORIES "${3rdlib_DIR}/onnxruntime/include")

# # We require CUDA, OpenCV, and TensorRT
# FIND_PACKAGE(TensorRT REQUIRED)
# FIND_PACKAGE(CUDA REQUIRED)
# FIND_PACKAGE(OpenCV REQUIRED)


SET(SAMPLE_SOURCES
    ./common/getopt.c
    ./common/sampleEngines.cpp
    ./common/logger.cpp
    ./common/sampleInference.cpp
    ./common/sampleOptions.cpp
    ./common/sampleReporting.cpp
    ./common/sampleReporting.cpp
	./common/sampleUtils.cpp
)
ADD_EXECUTABLE(${PROJECT_NAME} ${SAMPLE_SOURCES} ./trtexec.cpp)

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${CUDA_INCLUDE_DIRS}
    PUBLIC ${TensorRT_INCLUDE_DIR}
    PRIVATE ./common
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} libonnxruntime ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "52;61;75;86")
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

IF (MSVC)
	file(GLOB ORT_DLLS "${3rdlib_DIR}/onnxruntime/lib/*.dll")
	add_custom_command(TARGET ${PROJECT_NAME}
						POST_BUILD
						COMMAND ${CMAKE_COMMAND} -E copy_if_different
						${ORT_DLLS}
						$<TARGET_FILE_DIR:${PROJECT_NAME}>)
ENDIF (MSVC)

#===================================================================
ADD_EXECUTABLE(ORT2TRT ${SAMPLE_SOURCES} ./onnx2trt.cpp ./onnx2engine.cpp)
TARGET_INCLUDE_DIRECTORIES(ORT2TRT
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${CUDA_INCLUDE_DIRS}
    PUBLIC ${TensorRT_INCLUDE_DIR}
    PRIVATE ./common
)
TARGET_LINK_LIBRARIES(ORT2TRT libonnxruntime ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
SET_TARGET_PROPERTIES(ORT2TRT PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
SET_TARGET_PROPERTIES(ORT2TRT PROPERTIES CUDA_ARCHITECTURES "52;61;75;86")
SET_TARGET_PROPERTIES(ORT2TRT PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")