@echo off
echo Building FFNet.exe ...
g++ -std=c++11 -Wall -o FFNet.exe -I"../include" "../examples/FFNet Construction.cpp" "../src/ffnet.cpp" "../src/net.cpp"
echo Done
