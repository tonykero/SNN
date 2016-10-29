@echo off
echo Building FFNet-GeneticTrainer-O3.exe
g++ -std=c++11 -O3 -ffast-math -D__NO_INLINE__ -Wall -o FFNet-GeneticTrainer-O3.exe -I"../include" "../examples/FFNet & GeneticTrainer.cpp" "../src/ffnet.cpp" "../src/net.cpp" "../src/GeneticTrainer.cpp"
echo Done