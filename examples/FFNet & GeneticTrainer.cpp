#include "../include/snn.hpp"

#include <iostream>
#include <chrono>

std::vector<snn::Dataset> xor_ds = 
{
	snn::Dataset( {0.0f, 0.0f}, { 0.0f}),
	snn::Dataset( {1.0f, 0.0f}, { 1.0f}),
	snn::Dataset( {0.0f, 1.0f}, { 1.0f}),
	snn::Dataset( {1.0f, 1.0f}, { 0.0f}),
};

float evaluate(snn::Net net)
{
	float totalError = 0.0f;

	float output;

	for (unsigned int i = 0; i < 4; i++)
	{
		output = net.feed(xor_ds[i].inputs)[0];
		totalError += pow( xor_ds[i].outputs[0] - output , 2);
	}
	
	return 1/ totalError;
}

int main()
{
	std::cout << "-----FFNet & GeneticTrainer.exe----\n";
	std::cout << "Basic FFNet With:\n";
	std::cout << "L1: 2 Inputs, 1 bias\n";
	std::cout << "L2: 2 Hiddens, 1 bias\n";
	std::cout << "L3: 1 Output\n";
	std::cout << "Genetic Algorithm\n";
	std::cout << "Weight Initialisation:\n";
	std::cout << "range -1, 1\n";

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	std::vector<snn::Layer> layers =
	{
		snn::Layer(2, snn::NType::INPUT, 1),//2 Inputs, 1 BIAS
		snn::Layer(2, snn::NType::HIDDEN, 1),//2 Hiddens, 1 BIAS
		snn::Layer(1, snn::NType::OUTPUT)//1 Output
	};

	snn::FFNet net(layers);


	snn::GeneticTrainer gt(50, 0.1f, 0.5f, 25);
	gt.run(&net, 300, evaluate);

	std::cout << "----------Simple XOR Test----------\n";
	std::cout << "[0.0f, 0.0f] outputs [" << net.feed(xor_ds[0].inputs)[0] << "]\n";
	std::cout << "[1.0f, 0.0f] outputs [" << net.feed(xor_ds[1].inputs)[0] << "]\n";
	std::cout << "[0.0f, 1.0f] outputs [" << net.feed(xor_ds[2].inputs)[0] << "]\n";
	std::cout << "[1.0f, 1.0f] outputs [" << net.feed(xor_ds[3].inputs)[0] << "]\n";
	std::cout << "Global MSE (average): " << net.computeGlobalMSE(xor_ds) << "\n";
	std::cout << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()  << " ms)" << std::endl;
	return 0;
}