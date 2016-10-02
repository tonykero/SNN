#include "ffnet.hpp"
#include <random>

using namespace snn;
/*****************************Neuron**************************/
void Neuron::store(float val)
{

}

float Neuron::compute()
{

}
/*****************************Layer***************************/

Layer::Layer(unsigned int neuronsCount, unsigned int layerType)
{
    m_neuronsCount = neuronsCount;
    m_type = layerType;

    for(unsigned int i = 0; i < m_neuronsCount; i++)//fill the neurons array
    {
        Neuron n;
        n.type = m_type;
        n.id = i;
        n.parent = this;
        n.value = 0.0f;

        m_neurons.push_back(n);
    }

}

Layer::~Layer()
{

}

void Layer::setActivationFunction(std::function<float(float)> actFun)
{
    m_actFun = actFun;
}

std::vector<float> Layer::feed(std::vector<float> inputs)
{
    //TODO
}

std::vector<Neuron>* Layer::accessNeurons()
{
    return &m_neurons;
}

/*****************************FFNet***************************/

FFNet::FFNet(std::vector<unsigned int> layers)
{
    //create the layers
    for(unsigned int i = 0; i < layers.size(); i++)
    {   
        if(i != 0 && i != layers.size())
            Layer l(layers[i], layersType::HIDDEN);
        else if(i == 0)
            Layer l(layers[i], layersType::INPUT);
        else if(i == layers.size())
            Layer l(layers[i], layersType::OUTPUT);
        
        l.m_id = i;

        m_layers.push_back(l);
    }

    this->link();

}

FFNet::~FFNet()
{

}

std::vector<float> FFNet::feed(std::vector<float> inputs)
{

}

void FFNet::setOutputFunction(std::function<float(float)> actFun)
{
    m_outFun = actFun;
}

void FFNet::setHiddenFunction(std::function<float(float)> actFun)
{
    m_hidFun = actFun;
}

void FFNet::randWeights()
{

}

void FFNet::setLinks(std::vector<Link> links)
{
    m_links = links;
}

std::vector<Link> FFNet::getLinks()
{
    return m_links;
}

void FFNet::link()
{
    
}

std::vector<Layer>* FFNet::accessLayers()
{
    return &m_layers;
}