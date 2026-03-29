# 1. Minimum version and Project Name (Only define these ONCE)
cmake_minimum_required(VERSION 3.15)
project(ChainNano LANGUAGES CXX)
set(CMAKE_CUDA_ARCHITECTURES 89)

# 2. Enable Hot Reload for MSVC compilers (From your VS Template)
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# 3. Set C++ Standard globally
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# 4. Find LibTorch (vcpkg handles this)
find_package(Torch REQUIRED)

# 5. Define your executable and list ALL your .cpp files
add_executable(ChainNano
    Nano/Gumbel_MCTS.cpp
    Nano/InferenceQueue.cpp
    Nano/main.cpp
    Nano/MCTS.cpp
    Nano/ChainReaction.cpp
)

# 6. Tell CMake where to look for your .hpp and .h files
target_include_directories(ChainNano PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/Include 
    ${CMAKE_CURRENT_SOURCE_DIR}/Nano
)

# 7. Link LibTorch to your executable
target_link_libraries(ChainNano PRIVATE ${TORCH_LIBRARIES})

# 8. OS-Specific Performance Flags
if(MSVC)
    # Windows RTX 4060 build flags
    target_compile_options(ChainNano PRIVATE /W4 /MP /openmp)
else()
    # WSL Linux optimized build flags
    target_compile_options(ChainNano PRIVATE -Wall -Wextra -O3 -pthread -ffast-math)
endif()

if(MSVC)
    # Find all the DLLs in your LibTorch installation
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll" "${TORCH_INSTALL_PREFIX}/bin/*.dll")

    message(STATUS "---------------------------------------------------")
message(STATUS "CHECKING DLL PATH: Grabbing LibTorch files from -> ${TORCH_INSTALL_PREFIX}")
message(STATUS "---------------------------------------------------")

    # Create a post-build copy command for every single DLL
    foreach(DLL IN LISTS TORCH_DLLS)
        add_custom_command(TARGET ChainNano POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                "${DLL}" 
                $<TARGET_FILE_DIR:ChainNano>
        )
    endforeach()
endif()

# 9. --- POST-BUILD SCRIPT ---
# This automatically copies your model and web files to the output folder 
# so your executable doesn't crash trying to find them!
add_custom_command(TARGET ChainNano POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${CMAKE_CURRENT_SOURCE_DIR}/Nano/chain_reaction_bf16.pt" 
        $<TARGET_FILE_DIR:ChainNano>
)