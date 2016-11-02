/* Neural Network w/ Backprop training -> XOR function
* 2 layers feedforward neural network
* 2 inputs 1 BIAS, 2 Hidden 1 Bias, 1 Output
*/

#include "../include/net.hpp"
#include <iostream>

int main()
{
	std::vector<snn::Neuron> neurons;
	std::vector<snn::Link> links;
	std::vector<float> inputs;

	/*Neurons defs*/
	snn::Neuron I1;
	I1.type = snn::NType::INPUT;
	I1.id = 0;

	snn::Neuron I2;
	I2.type = snn::NType::INPUT;
	I2.id = 1;

	snn::Neuron B1;
	B1.type = snn::NType::BIAS;
	B1.id = 2;
	B1.output = 1;

	snn::Neuron H1;
	H1.type = snn::NType::HIDDEN;
	H1.id = 3;

	snn::Neuron H2;
	H2.type = snn::NType::HIDDEN;
	H2.id = 4;

	snn::Neuron B2;
	B2.type = snn::NType::BIAS;
	B2.id = 5;
	B2.output = 1;

	snn::Neuron O1;
	O1.type = snn::NType::OUTPUT;
	O1.id = 6;
	/*****/

	//store neurons in vector
	neurons.push_back(I1);//0
	neurons.push_back(I2);//1
	neurons.push_back(B1);//2
	neurons.push_back(H1);//3
	neurons.push_back(H2);//4
	neurons.push_back(B2);//5
	neurons.push_back(O1);//6

	/*Links defs*/
	snn::Link w1;
	w1.a = 0;//I1
	w1.b = 3;//H1
	w1.weight = -0.07f;

	snn::Link w2;
	w2.a = 0;//I1
	w2.b = 4;//H2
	w2.weight = 0.94f;

	snn::Link w3;
	w3.a = 1;//I2
	w3.b = 3;//H1
	w3.weight = 0.22f;

	snn::Link w4;
	w4.a = 1;//I2
	w4.b = 4;//H2
	w4.weight = 0.46f;

	snn::Link w5;
	w5.a = 2;//B1
	w5.b = 3;//H1
	w5.weight = -0.46f;

	snn::Link w6;
	w6.a = 2;//B1
	w6.b = 4;//H2
	w6.weight = 0.10f;

	snn::Link w7;
	w7.a = 3;//H1
	w7.b = 6;//O1
	w7.weight = -0.22f;

	snn::Link w8;
	w8.a = 4;//H2
	w8.b = 6;//O1
	w8.weight = 0.58f;

	snn::Link w9;
	w9.a = 5;//B2
	w9.b = 6;//O1
	w9.weight = 0.78f;
	/****/

	links.push_back(w1);
	links.push_back(w2);
	links.push_back(w3);
	links.push_back(w4);
	links.push_back(w5);
	links.push_back(w6);
	links.push_back(w7);
	links.push_back(w8);
	links.push_back(w9);

	snn::Net net(neurons, links);

	inputs.push_back(1.0f);
	inputs.push_back(0.0f);
	std::cout << "**********MAIN**********\n";
	std::cout << net.feed(inputs)[0];



    return 0;
}
