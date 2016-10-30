#include "../include/ffnet.hpp"

#include <iostream>
#include <chrono>

int main()
{
	std::cout << "-------------FFNet.exe-------------\n";
	std::cout << "Basic FFNet With:\n";
	std::cout << "L1: 2 Inputs, 1 bias\n";
	std::cout << "L2: 2 Hiddens, 1 bias\n";
	std::cout << "L3: 1 Output\n";
	std::cout << "No Training involved\n";
	std::cout << "Weight Initialisation:\n";
	std::cout << "range -1, 1\n";

	std::vector<snn::Dataset> xor_ds =
	{
		snn::Dataset({ 0.0f, 0.0f },{ 0.0f }),
		snn::Dataset({ 1.0f, 0.0f },{ 1.0f }),
		snn::Dataset({ 0.0f, 1.0f },{ 1.0f }),
		snn::Dataset({ 1.0f, 1.0f },{ 0.0f }),
	};

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	std::vector<snn::Layer> layers =
	{
		snn::Layer(2, snn::NType::INPUT, 1),//2 Inputs, 1 BIAS
		snn::Layer(2, snn::NType::HIDDEN, 1),//2 Hiddens, 1 BIAS
		snn::Layer(1, snn::NType::OUTPUT)//1 Output
	};
	
	snn::FFNet net(layers);
	net.randWeights(-1, 1);

	std::cout << "----------Simple XOR Test----------\n";
	std::cout << "[0.0f, 0.0f] outputs [" << net.feed( xor_ds[0].inputs )[0] << "]\n";
	std::cout << "[1.0f, 0.0f] outputs [" << net.feed( xor_ds[1].inputs )[0] << "]\n";
	std::cout << "[0.0f, 1.0f] outputs [" << net.feed( xor_ds[2].inputs )[0] << "]\n";
	std::cout << "[1.0f, 1.0f] outputs [" << net.feed( xor_ds[3].inputs )[0] << "]\n";
	std::cout << "Global MSE (average): " << net.computeGlobalMSE(xor_ds) << "\n";
	std::cout << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()  << " ms)" << std::endl;
	
	return 0;
}