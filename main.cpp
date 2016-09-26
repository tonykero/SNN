/* Neural Network w/ Backprop training -> XOR function
* Full Drogue
*/

#include "snn.hpp"
#include <vector>

int main()
{

    snn::FFNet net(2, 4, 1, 1);

    snn::TrainingMan trainM(snn::BACKPROP);

    trainM.train(&net, 
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
