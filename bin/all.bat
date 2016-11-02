@echo off
echo Building FFNet.exe ...
g++ -DNDEBUG -std=c++11 -Wall -o FFNet.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp"
echo Done
echo Building FFNet-DEBUG.exe ...
g++ -std=c++11 -Wall -o FFNet-DEBUG.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp"
echo Done
echo ------------------------------------
echo Building FFNet-O3.exe
g++ -DNDEBUG -std=c++11 -O3 -ffast-math -D__NO_INLINE__ -Wall -o FFNet-O3.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp" 
echo Done
echo Building FFNet-O3-DEBUG.exe
g++ -std=c++11 -O3 -ffast-math -D__NO_INLINE__ -Wall -o FFNet-O3-DEBUG.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp" 
echo Done
echo ------------------------------------
echo Building FFNet-GeneticTrainer.exe
g++ -DNDEBUG -std=c++11 -Wall -o FFNet-GeneticTrainer.exe -I"../include" "../examples/FFNet & GeneticTrainer.cpp" "../src/ffnet.cpp" "../src/net.cpp" "../src/GeneticTrainer.cpp"
echo Done
echo Building FFNet-GeneticTrainer-DEBUG.exe
g++ -std=c++11 -Wall -o FFNet-GeneticTrainer-DEBUG.exe -I"../include" "../examples/FFNet & GeneticTrainer.cpp" "../src/ffnet.cpp" "../src/net.cpp" "../src/GeneticTrainer.cpp"
echo Done
echo ------------------------------------
echo Building FFNet-GeneticTrainer-O3.exe
g++ -DNDEBUG -std=c++11 -O3 -ffast-math -D__NO_INLINE__ -Wall -o FFNet-GeneticTrainer-O3.exe -I"../include" "../examples/FFNet & GeneticTrainer.cpp" "../src/ffnet.cpp" "../src/net.cpp" "../src/GeneticTrainer.cpp"
echo Done
echo Building FFNet-GeneticTrainer-O3-DEBUG.exe
g++ -std=c++11 -O3 -ffast-math -D__NO_INLINE__ -Wall -o FFNet-GeneticTrainer-O3-DEBUG.exe -I"../include" "../examples/FFNet & GeneticTrainer.cpp" "../src/ffnet.cpp" "../src/net.cpp" "../src/GeneticTrainer.cpp"
echo Done