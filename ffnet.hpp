#pragma once

#include <vector>
#include <functional>
#include <string>

#include "actFunctions.hpp"


namespace snn
{

    enum NType: unsigned int
    {

        INPUT = 1,
        HIDDEN,
        OUTPUT,
        BIAS
        
    };

    struct Link
    {
        Neuron* a;
        Neuron* b;

        float weight = 1.0f;
    };

    struct Bias 
    {
        float output;
        unsigned int num;
        std::string id;
    };

    struct Neuron
    {
        float output;
        unsigned int type;
        std::string id;
        unsigned int num;
        //compute & store for future multi threading implementation
        float compute();
        void store(float value);

        private:
            float cacheValue;
    };

    class FFNet
    {
        public:
            FFNet(std::string script);

            ~FFNet();

            std::vector<float> feed(std::vector<float> inputs);

            void setOutputFunction(std::function<float(float)> actFun);
            void setHiddenFunction(std::function<float(float)> actFun);

            void randWeights();
            void setLinks(std::vector<Link>);
            std::vector<Link> getLinks();

            void link(std::string idA, std::string idB);
            void unlink(std::string idA, std::string idB);
            void add(unsigned int type);
            void remove(std:string id);

        private:

            std::map<std::string,Neuron> m_neurons;

            std::vector<Neuron*> m_inputs;
            std::vector<Neuron*> m_hiddens;
            std::vector<Neuron*> m_outputs;
            std::vector<Neuron*> m_bias;

            std::vector<Link> m_links;

            std::function<float(float)> m_hidFun = snn::FUN_SIGMOID;
            std::function<float(float)> m_outFun = snn::FUN_SIGMOID;

            
    };

}

