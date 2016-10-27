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
#include <iostream>

using namespace snn;

/*****************************Layer***************************/
Layer::Layer(unsigned int neuronsCount, unsigned int neuronsType, unsigned int biasCount)
{
    for(unsigned int i = 0; i < neuronsCount; i++)
    {
        Neuron n;
        n.type = neuronsType;
        neurons.push_back(n);
    }


    if(biasCount > 0)
        for(unsigned int i = 0; i < biasCount; i++)
        {
            Neuron n;
            n.type = NType::BIAS;
            n.output = 1.0f;
            neurons.push_back(n);
        }
    
    type = neuronsType;
}
/*****************************Neuron**************************/
float Neuron::compute()
{
    switch(type)
    {
        case NType::INPUT:
            output = parent->m_inputs[id];
            break;
        case NType::HIDDEN:
            output = parent->m_hidFun(sum);
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

Net::Net(std::vector<Neuron> neurons, std::vector<Link> links)
{
    m_neurons = neurons;
    m_neuronsCount = neurons.size();

    m_links = links;
	m_linksCount = links.size();

    for(unsigned int i = 0; i < m_neuronsCount-1; i++)
    {
		m_neurons[i].id = i;
        m_neurons[i].parent = this;
    }
}

Net::~Net()
{

}

std::vector<float> Net::feed(std::vector<float> inputs)
{
    m_inputs = inputs;

    unsigned int actualID = 1; //the first neuron can't have 1 as id


    //m_links: feedforward links, and stored by order
    //When in the list the previous neuron is not the same as the actual
    //we compute it, because we know that there are no dependencies in the next neurons
    for(unsigned int i = 0; i < m_linksCount; i++)
    {
		unsigned int idA = m_links[i].a;
		unsigned int idB = m_links[i].b;
		float weight = m_links[i].weight;
		if(actualID != m_neurons[idA].id)//if actualneuron != link.a
        {
			m_neurons[idB].sum += m_neurons[idA].compute()*weight;
        }
        else //if actualneuron == link.a
        {
            m_neurons[idB].sum += m_neurons[idA].output*weight;//we must not recompute the same neuron
        }
        actualID = m_neurons[idA].id;
    }

    //find all outputs neurons, compute them, return them w/ outputs vector
    std::vector<float> outputs;
    for(unsigned int i = 0; i < m_neuronsCount; i++)
    {
        if(m_neurons[i].type == NType::OUTPUT)
        {
            outputs.push_back(m_outFun(m_neurons[i].sum));
        }
    }

    return outputs;

}

void Net::setOutputFunction(std::function<float(float)> actFun)
{
    m_outFun = actFun;
}

void Net::setHiddenFunction(std::function<float(float)> actFun)
{
    m_hidFun = actFun;
}

std::function<float(float)> Net::getOutputFunction()
{
    return m_outFun;
}
std::function<float(float)> Net::getHiddenFunction()
{
    return m_hidFun;
}


unsigned int Net::getSize()
{
    return m_neurons.size();
}

void Net::randWeights()
{
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distrib(-1, 1);
	for (unsigned int i = 0; i < m_links.size(); i++)
	{
		m_links[i].weight = distrib(generator);
	}
}

void Net::setLinks(std::vector<Link> links)
{
    m_links = links;
}

std::vector<Link> Net::getLinks()
{
    return m_links;
}

std::vector<Neuron> Net::getNeurons()
{
    return m_neurons;
}


void Net::computeDeltas(std::vector<float> expectedOutputs, std::vector<float> observedOutputs)
{
    //this function is called right after a feed() call on the net 

	//reset initial values
	for (unsigned int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].isDeltaComputed = false;
	}

    //compute output deltas
    unsigned int firstOutput = 0;
    for(unsigned int i = m_neurons.size()-1; i > 0; i--)
    {

        if(m_neurons[i].type == NType::OUTPUT)
        {
            m_neurons[i].delta = (expectedOutputs[m_neurons.size() - 1 - i]-observedOutputs[m_neurons.size() - 1 - i]) * m_outFun_derivative(observedOutputs[m_neurons.size() - 1 - i]);
			m_neurons[i].isDeltaComputed = true;
        }
        else
        {
            break;
        }

    }

    //compute interior deltas
    //TODO: compute interior deltas
    //according to the formula
    //Dy = derivative * sigma (Dk*Wkj)
    //calculate Dy for each neurons

    for(unsigned int i = m_neurons.size()-1; i > 0; i--)
    {
        if( m_neurons[i].type == NType::HIDDEN)
        {
            float sum = 0.0f;
            if (m_neurons[i].isDeltaComputed == false)
				{
					//m_neurons[i] is not a computed neuron 
					for (unsigned int k = m_links.size() - 1; k > 0; k--)
					{
						if (m_links[k].a == m_neurons[i].id)
						{
							
								if (m_neurons[m_links[k].b].isDeltaComputed == true)
								{
									sum += m_neurons[m_links[k].b].delta * m_links[k].weight;
								}
						}
					}

					m_neurons[i].delta = m_hidFun_derivative(m_neurons[i].sum) * sum;
					m_neurons[i].isDeltaComputed = true;

				}
				else
					break;
        }
		
        
    }

}