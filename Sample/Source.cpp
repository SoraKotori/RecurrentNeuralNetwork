#include "RecurrentNeuralNetwork.h"
#include <cstdlib>
#include <iostream>
#include <iterator>

using namespace NeuralNetwork;
using namespace std;

int main()
{
    using _Type = double;
    using _IntType = char;
    size_t _TimeCount = 10;
    size_t _Dimension = std::numeric_limits<_IntType>::max();
    size_t _LayerCount = 20;

    VectorType<_IntType> _Sequence(_TimeCount);
    VectorType<VectorType<_Type>> _Input(_TimeCount, VectorType<_Type>(_Dimension));
    std::iota(std::begin(_Sequence), std::end(_Sequence), _IntType(50));
    std::copy(std::begin(_Sequence), std::end(_Sequence), std::ostream_iterator<unsigned>(std::cout, " "));
    IntToOnehot(std::begin(_Sequence), std::end(_Sequence), std::begin(_Input));
    cout << endl << endl;

    //OnehotToInt(std::begin(_Input), std::end(_Input), std::begin(_Sequence));
    //std::copy(std::begin(_Sequence), std::end(_Sequence), std::ostream_iterator<unsigned>(std::cout, " "));
    //cout << endl << endl;

    _Type _LearningRate = 0.2;
    int _IteratorCount = 1000;
    RecurrentNeuralNetwork<_Type> MyRNN(_TimeCount, _Dimension, _LayerCount, _LearningRate);

    MyRNN.ForwardPropagation(_Input);
    auto&& _Output = MyRNN.GetOutput();
    OnehotToInt(std::begin(_Output), std::end(_Output), std::begin(_Sequence));
    std::copy(std::begin(_Sequence), std::end(_Sequence), std::ostream_iterator<unsigned>(std::cout, " "));
    cout << endl << endl;

    //std::for_each(std::begin(_Output), std::end(_Output), [](auto&& _Vector)
    //{
    //    std::copy(std::begin(_Vector), std::end(_Vector), std::ostream_iterator<_Type>(std::cout, " "));
    //    cout << endl;
    //});
    //cout << endl << endl;

    //cout << MyRNN.GetOutput() << endl;

    int _Iterator = 0;
    for (; _Iterator < _IteratorCount; ++_Iterator)
    {
        cout << "_Iterator: " << _Iterator << endl;

        MyRNN.ForwardPropagation(_Input);
        MyRNN.BackPropagationThroughTime(_Input, _Input);

        auto&& _Output = MyRNN.GetOutput();
        OnehotToInt(std::begin(_Output), std::end(_Output), std::begin(_Sequence));
        std::copy(std::begin(_Sequence), std::end(_Sequence), std::ostream_iterator<unsigned>(std::cout, " "));
        cout << endl << endl;
    }
    cout << "_Iterator: " << _Iterator << endl;

    MyRNN.ForwardPropagation(_Input);
    auto&& _Output2 = MyRNN.GetOutput();
    OnehotToInt(std::begin(_Output), std::end(_Output), std::begin(_Sequence));
    std::copy(std::begin(_Sequence), std::end(_Sequence), std::ostream_iterator<unsigned>(std::cout, " "));
    cout << endl << endl;

    return EXIT_SUCCESS;
}