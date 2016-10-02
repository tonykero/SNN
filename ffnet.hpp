#pragma once

#include <vector>
#include <functional>
#include <string>

#include "actFunctions.hpp"


namespace snn
{

    enum neuronTypes: unsigned int
    {

        INPUT = 1,
        HIDDEN,
        OUTPUT,
        BIAS
        
    };

    enum layerTypes: unsigned int
    {
        INPUT = 1,
        HIDDEN,
        OUTPUT
    };

    struct Link
    {
        Neuron* a;
        Neuron* b;

        float weight = 1.0f;
    };

    struct Neuron
    {
        Layer* parent;
        float value;
        unsigned int type;
        unsigned int id;

        //compute & store for future multi threading implementation
        float compute();
        void store(float value);

        private:
            float cacheValue;
    };

    class Layer
    {
        public:

            Layer(unsigned int neuronsCount, unsigned int layerType);
            ~Layer();

            std::vector<float> feed(std::vector<float> inputs);

            void setActivationFunction(std::function<float(float)> actFun);

            std::vector<Neuron>* accessNeurons();

        private:
            std::vector<Neuron> m_neurons;
            unsigned int m_neuronsCount = 0;

            unsigned int m_type;
            unsigned int m_id;

            std::function<float(float)> m_actFun = snn::FUN_SIGMOID;

        friend float Neuron::compute(); //for m_actFun
        friend class FFNet;
    };

    class FFNet
    {
        public:
            FFNet(std::vector<unsigned int> layers);
            ~FFNet();

            std::vector<float> feed(std::vector<float> inputs);

            void setOutputFunction(std::function<float(float)> actFun);
            void setHiddenFunction(std::function<float(float)> actFun);

            void randWeights();
            void setLinks(std::vector<Link>);
            std::vector<Link> getLinks();
            void link();

            std::vector<Layer>* accessLayers();

        private:

            std::vector<Layer> m_layers;
            std::vector<Link> m_links;

            std::function<float(float)> m_hidFun = snn::FUN_SIGMOID;
            std::function<float(float)> m_outFun = snn::FUN_SIGMOID;

            
    };

}

