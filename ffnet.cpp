#include "ffnet.hpp"
#include <random>

using namespace snn;
/*****************************Neuron**************************/
void Neuron::store(float val)
{
    //TODO: store()
}

float Neuron::compute()
{
    //TODO: compute()
}

/*****************************FFNet***************************/

FFNet::FFNet(std::string script)
{

}

FFNet::~FFNet()
{

}

std::vector<float> FFNet::feed(std::vector<float> inputs)
{
    //TODO: FFnet::feed()
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
    //TODO: randWeights()
}

void FFNet::setLinks(std::vector<Link> links)
{
    m_links = links;
}

std::vector<Link> FFNet::getLinks()
{
    return m_links;
}

void FFNet::link(std::string idA, std::string idB)
{

}

void FFNet::unlink(std::string idA, std::string idB)
{

}

void FFNet::add(unsigned int type)
{

}

void FFNet::remove(std:string id)
{

}