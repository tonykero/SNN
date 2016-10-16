/* Neural Network w/ Backprop training -> XOR function
* 2 layers feedforward neural network 
* 2 inputs 1 BIAS, 2 Hidden 1 Bias, 1 Output
*/

#include "snn.hpp"
#include <vector>

int main()
{

    /*outdated thoughts*/
    
    //snn::Net net();

    snn::Trainer tr(snn::BACKPROP);

    tr.train(&net, 
                    {
                        snn::Dataset({0,0},{0}),
                        snn::Dataset({0, 1},{1}),
                        snn::Dataset({1, 0},{1}),
                        snn::Dataset({1, 1},{0})
                }
                );

    std::cout << net.feed({0, 0})[0];

    return 0;
}
