@echo off
echo Building FFNet-O3.exe
g++ -DNDEBUG -std=c++11 -O3 -ffast-math -D__NO_INLINE__ -Wall -o FFNet-O3.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp" 
echo Done
echo Building FFNet-O3-DEBUG.exe
g++ -std=c++11 -O3 -ffast-math -D__NO_INLINE__ -Wall -o FFNet-O3-DEBUG.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp" 
echo Done