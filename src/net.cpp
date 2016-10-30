/*********************************************************************
*Copyright (C) 2016  Antoine Karcher				                 *
*								                                     *
*This program is free software: you can redistribute it and/or modify*
*it under the terms of the GNU General Public License as published by*
*the Free Software Foundation, either version 3 of the License, or   *
*(at your option) any later version.				                 *
*								                                     *
*This program is distributed in the hope that it will be useful,     *
*but WITHOUT ANY WARRANTY; without even the implied warranty of	     *
*MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the       *
*GNU General Public License for more details.			             *
*                                                                    *
*You should have received a copy of the GNU General Public License   *
*along with this program. If not, see <http://www.gnu.org/licenses/>.*
*********************************************************************/

#include "../include/net.hpp"
#include <random>

using namespace snn;
/*****************************Layer***************************/
Layer::Layer(unsigned int _neuronsCount, unsigned int _neuronsType, unsigned int _biasCount)
{
    for(unsigned int i = 0; i < _neuronsCount; i++)
    {
        Neuron n;
        n.type = _neuronsType;
        neurons.push_back(n);
    }


    if(_biasCount > 0)
        for(unsigned int i = 0; i < _biasCount; i++)
        {
            Neuron n;
            n.type = NType::BIAS;
            n.output = 1.0f;
            neurons.push_back(n);
        }
    
    type = _neuronsType;
}
/*****************************Neuron**************************/
float Neuron::compute(std::vector<float> _inputs)
{
    switch(type)
    {
        case NType::INPUT:
            output = _inputs[id];
            break;
        case NType::HIDDEN:
            output = parent->hidFun_m(sum);
            break;
        case NType::BIAS:
            //output already defined
            break;
		/*
        case NType::OUTPUT:
            output = parent->m_outFun(output);
            break;*/
    }
    
	return output;
}

/*******************************Net***************************/
Net::Net()
{

}

Net::Net(std::vector<Neuron> _neurons, std::vector<Link> _links)
{
    neurons_m = _neurons;
    neuronsCount_m = _neurons.size();

    links_m = _links;
	linksCount_m = _links.size();

    for(unsigned int i = 0; i < neuronsCount_m-1; i++)
    {
		neurons_m[i].id = i;
        neurons_m[i].parent = this;
    }
}

Net::~Net()
{

}

std::vector<float> Net::feed(std::vector<float> _inputs)
{
    inputs_m = _inputs;

    unsigned int actualID = 1; //the first neuron can't have 1 as id

    //m_links: feedforward links, and stored by order
    //When in the list the previous neuron is not the same as the actual
    //we compute it, because we know that there are no dependencies in the next neurons
    for(unsigned int i = 0; i < linksCount_m; i++)
    {
		unsigned int idA = links_m[i].a;
		unsigned int idB = links_m[i].b;
		float weight = links_m[i].weight;

        if(actualID != neurons_m[idA].id)//if actualneuron != link.a
        {
			neurons_m[idB].sum += neurons_m[idA].compute(_inputs)*weight;
        }
        else //if actualneuron == link.a
        {
            neurons_m[idB].sum += neurons_m[idA].output*weight;//we must not recompute the same neuron
        }
        actualID = neurons_m[idA].id;
    }

    //find all outputs neurons, compute them, return them w/ outputs vector
    std::vector<float> outputs;
    for(unsigned int i = 0; i < neuronsCount_m; i++)
    {
        if(neurons_m[i].type == NType::OUTPUT)
        {
            neurons_m[i].output = outFun_m(neurons_m[i].sum);
            outputs.push_back( neurons_m[i].output );
        }
    }

    for(unsigned int i = 0; i < neurons_m.size(); i++)
    {
        neurons_m[i].sum = 0.0f;
    }

    return outputs;

}

float Net::computeMSE(std::vector<float> _inputs, std::vector<float> _outputs)
{
	std::vector<float> outputs = this->feed(_inputs);

	float error = 0.0f;
	for (unsigned int i = 0; i < outputs.size(); i++)
	{
		error += pow(_outputs[i] - outputs[i], 2);
	}

	return error / outputs.size();
}

float Net::computeGlobalMSE(std::vector<Dataset> _ds)
{
	float error = 0.0f;
	for (unsigned int i = 0; i < _ds.size(); i++)
	{
		error += computeMSE(_ds[i].inputs, _ds[i].outputs);
	}

	return error / _ds.size();
}

void Net::setOutputFunction(std::function<float(float)> _actFun)
{
    outFun_m = _actFun;
}

void Net::setHiddenFunction(std::function<float(float)> _actFun)
{
    hidFun_m = _actFun;
}

std::function<float(float)> Net::getOutputFunction()
{
    return outFun_m;
}
std::function<float(float)> Net::getHiddenFunction()
{
    return hidFun_m;
}


unsigned int Net::getSize()
{
    return neurons_m.size();
}

void Net::randWeights(float _a, float _b)
{
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distrib(_a, _b);
	for (unsigned int i = 0; i < links_m.size(); i++)
	{
		links_m[i].weight = distrib(generator);
	}
}

void Net::setLinks(std::vector<Link> _links)
{
    links_m = _links;
}

std::vector<Link> Net::getLinks()
{
    return links_m;
}

std::vector<Neuron> Net::getNeurons()
{
	return neurons_m;
}