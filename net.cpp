#include "net.hpp"
#include <random>

using namespace snn;
/*****************************Neuron**************************/
void Neuron::compute()
{
    switch(type)
    {
        case NType::INPUT:
            output = parent->m_inputs[id-1];
            break;
        case NType::HIDDEN:
            output = m_hidFun(output);
            break;
        case NType::BIAS:
            //output already defined
            break;
        case NType::OUTPUT:
            output = m_outFun(output);
            break;
    }
}

/*******************************Net***************************/

Net::Net(std::vector<Neuron> neurons, std::vector<Link> links)
{
    m_neurons = neurons;
    m_neuronsCount = neurons.size();

    m_links = links;

    for(unsigned int i = 0; i < m_neuronsCount; i++)
    {
        m_neurons[i].parent = this;
    }
}

Net::~Net()
{

}

std::vector<float> Net::feed(std::vector<float> inputs)
{
    m_inputs = inputs;

    actualType = NType::INPUT;
    actualID = 0; //impossible value for a normal neuron
    //we are sure that this identity cannot be the same at the first
    //processed neuron


    //m_links: feedforward links, and stored by order
    //When in the list the previous neuron is not the same as the actual
    //we compute it, because we know that there are no dependencies in the next neurons
    for(unsigned int i = 0; i < m_links.size(); i++)
    {
        if(actualType != m_links[i].a->type
            && actualID != m_links[i].a->id)//if actualneuron != link.a
        {
            m_links[i].b->output += m_links[i].a->compute();
        }
        else //if actualneuron == link.a
        {
            m_links[i].b->output += m_links[i].a->output;//no need to recompute
        }
        actualType = m_links[i].a->type;
        actualID = m_links[i].a->id;
    }

    //find all outputs neurons, compute them, return them w/ outputs vector
    std::vector<float> outputs;
    for(unsigned int i = 0; i < m_neurons.size(); i++)
    {
        if(m_neurons[i].type == NType::OUTPUT)
        {
            m_neurons[i].compute();
            outputs.push_back(m_neurons[i].output);
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

std::function<float(float)> getOutputFunction()
{
    return m_outFun;
}
std::function<float(float)> getHiddenFunction()
{
    return m_hidFun;
}


unsigned int Net::getSize()
{
    return m_neurons.size();
}

void Net::randWeights()
{
    //TODO: randWeights()
}

void Net::setLinks(std::vector<Link> links)
{
    m_links = links;
}

std::vector<Link> Net::getLinks()
{
    return m_links;
}