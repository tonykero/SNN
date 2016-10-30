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

#include "../include/GeneticTrainer.hpp"

#include <random>


using namespace snn;

GeneticTrainer::GeneticTrainer(unsigned int _populationPerGen, float _mutationRate, float _crossoverRate, unsigned int _eliteCopies)
{
    #ifdef DEBUG
    assert( _eliteCopies < _populationPerGen );
    assert( _mutationRate >= 0.0f && _mutationRate <= 1.0f );
    assert( _crossoverRate >= 0.0f && _crossoverRate <= 1.0f );
    #endif

    populationPerGen_m = _populationPerGen;
    mutationRate_m = _mutationRate;
    crossoverRate_m = _crossoverRate;
    eliteCopies_m = _eliteCopies;
}

GeneticTrainer::~GeneticTrainer()
{

}

void GeneticTrainer::run(Net *_net, unsigned int _generations, std::function<float(Net)> _fitnessFunction)
{
    
    //generate random population 
    
    //evaluate fitnesses of the population 
    //Selection
    //Crossover
    //Mutation
    //Accepting
    //Replace
    //back to second step

    fitnesses_m.clear();
    population_m.clear();

    model_m = _net->getLinks();

    //Generate initial random population
    for(unsigned int i = 0; i < populationPerGen_m; i++)
    {
        population_m.push_back(randomWeights());
    }
    
    std::default_random_engine generator;
    

    std::vector< std::vector<Link> > nextPopulation;

	unsigned int eliteIndex = 0;

    for(unsigned int i = 0; i < _generations; i++)
    {
        nextPopulation.clear();
		fitnesses_m.clear();

        //Evaluate population
        float totalFitness = 0;
        for(unsigned int j = 0; j < populationPerGen_m; j++)
        {
            _net->setLinks(population_m[j]);
            fitnesses_m.push_back( _fitnessFunction(*_net) );
            totalFitness += fitnesses_m.back();
        }

        //Store the best
		eliteIndex = 0;
        for(unsigned int j = 1; j < fitnesses_m.size(); j++)
        {
            if(fitnesses_m[j] > fitnesses_m[eliteIndex])
                eliteIndex = j;
        }

        //insert multiple times the fittest
        for(unsigned int j = 0; j < eliteCopies_m; j++)
        {
            nextPopulation.push_back(population_m[eliteIndex]);
        }

        std::uniform_int_distribution<unsigned int> distrib(0, (unsigned int)totalFitness);
		std::vector< std::vector<Link>> selected;

		//Roulette Wheel Selection
		//basically all candidates have the same probability of being picked
		//TODO: choose only those which have a fitness higher than the average
		while(nextPopulation.size() < populationPerGen_m )
        {

            unsigned int randLvl1 = distrib(generator);
			unsigned int randLvl2 = distrib(generator);
			float sum1 = 0.0f;
			float sum2 = 0.0f;

            for(unsigned int k = 0; k < fitnesses_m.size(); k++)
            {
                if(sum1 < (float)randLvl1)
                    sum1 += fitnesses_m[k];
				
				if((selected.size() != 1) && (sum1 >= (float)randLvl1))
					selected.push_back(population_m[k]);

				if (sum2 < (float)randLvl2)
					sum2 += fitnesses_m[k];
				
				if ( (selected.size() != 2 ) && ( sum2 >= (float)randLvl2 ) )
					selected.push_back(population_m[k]);
				
				if ( (sum1 >= (float)randLvl1 && sum2 >= (float)randLvl2 ))
					break;
            }

            
            //we have now 2 candidates
            //we need to do a crossover on them
            //then we need to mutate their offsprings

            std::vector<Link> candidate1 = selected[0];
            std::vector<Link> candidate2 = selected[1];

            selected.clear();

            std::vector<Link> offspring1 = candidate1,
								offspring2 = candidate2;

			
			for(unsigned int k = 0; k < candidate1.size(); k++)
			{
				//uniform crossover defined by crossoverRate
				std::uniform_int_distribution<unsigned int> crossDistrib(0, 100);
				if(crossDistrib(generator) < crossoverRate_m*candidate1.size() )
                {
					offspring1[k].weight = candidate2[k].weight;
                    offspring2[k].weight = candidate1[k].weight;
				}

				
				std::uniform_real_distribution<float> weightDistrib(-2, 2);
                
				//mutation
				//here the mutation consists of multiplying the mutated weight by a random number (-2, 2)

                if(crossDistrib(generator) <= mutationRate_m*100 )
                {
                    offspring1[k].weight *= weightDistrib(generator);
                }

                if(crossDistrib(generator) <= mutationRate_m*100 )
                {
                    offspring2[k].weight *= weightDistrib(generator);
                }
            }

			nextPopulation.push_back(offspring1);
			nextPopulation.push_back(offspring2);


        }
        population_m = nextPopulation;
    }

	_net->setLinks( population_m[eliteIndex] );

}

std::vector<Link> GeneticTrainer::randomWeights()
{
    //randomize weights according to model_m structure

	std::vector<Link> rC = model_m;

    std::default_random_engine gen;
    std::uniform_real_distribution<float> weightDistrib(-1, 1);
    for(unsigned int i = 0; i < rC.size(); i++)
    {
        rC[i].weight = weightDistrib(gen);
    }
	return rC;
}