@echo off
echo Building FFNet.exe ...
g++ -DNDEBUG -std=c++11 -Wall -o FFNet.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp"
echo Done
echo Building FFNet-DEBUG.exe ...
g++ -std=c++11 -Wall -o FFNet-DEBUG.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp"
echo Done