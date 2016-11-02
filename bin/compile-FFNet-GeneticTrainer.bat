@echo off
echo Building FFNet-GeneticTrainer.exe
g++ -DNDEBUG -std=c++11 -Wall -o FFNet-GeneticTrainer.exe -I"../include" "../examples/FFNet & GeneticTrainer.cpp" "../src/ffnet.cpp" "../src/net.cpp" "../src/GeneticTrainer.cpp"
echo Done
echo Building FFNet-GeneticTrainer-DEBUG.exe
g++ -std=c++11 -Wall -o FFNet-GeneticTrainer-DEBUG.exe -I"../include" "../examples/FFNet & GeneticTrainer.cpp" "../src/ffnet.cpp" "../src/net.cpp" "../src/GeneticTrainer.cpp"
echo Done