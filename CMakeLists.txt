cmake_minimum_required(VERSION 2.8.12)

project(SNN)

option(BUILD_SHARED "ON to build SNN as SHARED, OFF to build SNN AS STATIC" ON)
option(BUILD_EXAMPLES "Builds examples if ON" OFF)
option(DEBUG "Adds in-code features to help debugging" OFF)

cmake_policy(SET CMP0054 NEW)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    if(DEBUG)
        #Debug Mode
        add_compile_options(-std=c++11
                    -Wall
                    -Wextra
                    -pedantic
                    -g
                    )
    else()
        #Release Mode
        add_compile_options(-std=c++11
                    -Wall
                    -Wextra
                    -pedantic
                    -s
                    -O2
                    -D__NO_INLINE__
                    -DNDEBUG
                    )
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    if(DEBUG)
        #Debug Mode
        add_compile_options(-std=c++11
                    -Wall
                    -Wextra
                    -pedantic
                    -g
                    )
    else()
        #Release Mode
        add_compile_options(-std=c++11
                    -Wall
                    -Wextra
                    -pedantic
                    -s
                    -O2
                    -D__NO_INLINE__
                    -DNDEBUG
                    )
    endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
    if(DEBUG)
        #Debug Mode
        add_compile_options(/std=c++11
                    /W4
                    /Wall
                    )
    else()
        #Release Mode
        add_compile_options(/std=c++11
                    /W4
                    /Wall
                    /DNDEBUG
                    )
    endif()
endif()


set(SOURCES
    "include/Config.hpp"
    "include/net.hpp"
    "include/ffnet.hpp"
    "include/snn.hpp"
    "include/util.hpp"
    "include/GeneticTrainer.hpp"
    "include/BackpropTrainer.hpp"
    "src/net.cpp"
    "src/ffnet.cpp"
    "src/GeneticTrainer.cpp"
    "src/BackpropTrainer.cpp"
    )

if(BUILD_SHARED)
    add_library(SNN SHARED ${SOURCES})
else()
    add_library(SNN STATIC ${SOURCES})
endif()

if(BUILD_EXAMPLES)
    #FFNet Construction
    set(SOURCE
        "examples/FFNet Construction.cpp"
        )

    set(LINK_LIBS ${LINK_LIBS} SNN)
    add_executable(FFNet-Construction ${SOURCE})
    target_link_libraries(FFNet-Construction ${LINK_LIBS})

    #FFNet & GeneticTrainer
    set(SOURCE
        "examples/FFNet & GeneticTrainer.cpp"
        )

    add_executable(FFNet-GeneticTrainer ${SOURCE})
    target_link_libraries(FFNet-GeneticTrainer ${LINK_LIBS})

    #FFNet & BackpropTrainer
    set(SOURCE
        "examples/FFNet & BackpropTrainer.cpp"
        )

    add_executable(FFNet-BackpropTrainer ${SOURCE})
    target_link_libraries(FFNet-BackpropTrainer ${LINK_LIBS})
endif()