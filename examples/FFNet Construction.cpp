#include "../include/ffnet.hpp"

#include <iostream>

int main()
{
	std::vector<snn::Layer> layers =
	{
		snn::Layer(2, snn::NType::INPUT, 1),//2 Inputs, 1 BIAS
		snn::Layer(2, snn::NType::HIDDEN, 1),//2 Hiddens, 1 BIAS
		snn::Layer(1, snn::NType::OUTPUT)//1 Output
	};
	
	snn::FFNet net(layers);
	net.randWeights(-1, 1);

	std::cout << net.feed({ 1.0f, 0.0f })[0];

	return 0;
}