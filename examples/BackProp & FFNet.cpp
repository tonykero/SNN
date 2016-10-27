#include <iostream>

#include "../include/snn.hpp"

using namespace snn;

int main()
{
	std::vector<Layer> layers =
	{
		Layer(2, NType::INPUT, 1),//2 Inputs, 1 BIAS
		Layer(2, NType::HIDDEN, 1),//2 Hiddens, 1 BIAS
		Layer(1, NType::OUTPUT)//1 Output
	};
	
	FFNet net(layers);
	net.randWeights();

	

	std::vector<Dataset> ds =
	{
		Dataset({0.0f, 0.0f}, {0.0f}),
		Dataset({1.0f, 0.0f}, {1.0f}),
		Dataset({0.0f, 1.0f}, {1.0f}),
		Dataset({1.0f, 1.0f}, {0.0f})
	};

	
    BackProp bp(&net, DER_SIGMOID, DER_SIGMOID);
	bp.train(ds, 0.5, 0.1, 200);

	/*
	std::cout << "----NEURONS-----\n";
	std::vector<Neuron> neurons = net.getNeurons();
	for (unsigned int i = 0; i < neurons.size(); i++)
	{
		std::cout << "-----\n";
		std::cout << "i: " << i << "\n";
		std::cout << "ID: " << neurons[i].id;
		std::cout << "\nType: " << neurons[i].type << std::endl;
	}

	std::cout << "----LINKS----\n";
	std::vector<Link> links = net.getLinks();
	for (unsigned int i = 0; i < links.size(); i++)
	{
		std::cout << "-----\n";
		std::cout << "i: " << i << "\n";
		std::cout << "IDA: " << links[i].a << "\n";
		std::cout << "IDB: " << links[i].b << "\n";
		std::cout << "weight: " << links[i].weight << std::endl;
	}
	*/


	std::cout << "------------\n";
	std::cout << (float)net.feed({ 0.0f, 0.0f })[0];
	std::cout << "\n";
	std::cout << net.feed({ 1.0f, 0.0f })[0];
	std::cout << "\n";
	std::cout << net.feed({ 0.0f, 1.0f })[0];
	std::cout << "\n";
	std::cout << net.feed({ 1.0f, 1.0f })[0] << std::endl;

	std::vector<Link> links = net.getLinks();
	for (unsigned int i = 0; i < links.size(); i++)
	{
		std::cout << "\n----------\n";
		std::cout << "A: " << links[i].a << "\n"
			<< "B: " << links[i].b << "\n"
			<< "weight: " << links[i].weight;
	}
	return 0;
}