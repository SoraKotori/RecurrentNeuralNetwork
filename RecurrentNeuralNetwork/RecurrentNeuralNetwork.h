#pragma once
#include <boost\numeric\ublas\vector.hpp>
#include <boost\numeric\ublas\io.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <numeric>
#include <type_traits>
#include <utility>

namespace NeuralNetwork
{
    template<typename _Type>
    using VectorType = std::vector<_Type>;
    //using VectorType = boost::numeric::ublas::vector<_Type>;

    template<typename _ForwardIterator1, typename _ForwardIterator2>
    bool IntToOnehot(_ForwardIterator1 _First1, _ForwardIterator1 _Last1, _ForwardIterator2 _First2)
    {
        for (; _First1 != _Last1; ++_First1, ++_First2)
        {
            auto&& _Index = static_cast<size_t>(*_First1);
            if (0 > _Index)
            {
                return false;
            }
            auto&& _Vector = *_First2;

            std::fill(std::begin(_Vector), std::end(_Vector), 0);
            _Vector[_Index] = 1;
        }

        return true;
    }

    template<typename _ForwardIterator1, typename _ForwardIterator2>
    void OnehotToInt(_ForwardIterator1 _First1, _ForwardIterator1 _Last1, _ForwardIterator2 _First2)
    {
        for (; _First1 != _Last1; ++_First1, ++_First2)
        {
            auto&& _Vector = *_First1;
            auto&& _MaxIterator = std::max_element(std::begin(_Vector), std::end(_Vector));
            auto&& _MaxIndex = std::distance(std::begin(_Vector), _MaxIterator);

            *_First2 = _MaxIndex;
        }
    }

    template<typename _ForwardIterator>
    void Softmax(_ForwardIterator _First, _ForwardIterator _Last)
    {
        auto _Max = *max_element(_First, _Last);

        auto&& _Sum = accumulate(_First, _Last, decltype(_Max)(0), [_Max](auto&& _Init, auto&& _Value)
        {
            return _Init + exp(_Value - _Max);
        });

        transform(_First, _Last, _First, [_Denominator = exp(_Max + log(_Sum))](auto&& _Value)
        {
            return exp(_Value) / _Denominator;
        });
    }

    template<typename _ForwardIterator1, typename _ForwardIterator2, typename Function>
    void ForEach(_ForwardIterator1 _First1, _ForwardIterator1 _Last1, _ForwardIterator2 _First2, Function func)
    {
        for (; _First1 != _Last1; ++_First1, ++_First2)
        {
            func(*_First1, *_First2);
        }
    }

    template<typename _InputIterator1, typename _InputIterator2, typename _ForwardIterator>
    void OuterProduct(_InputIterator1 _First1, _InputIterator1 _Last1,
        _InputIterator2 _First2, _InputIterator2 _Last2, _ForwardIterator _Result)
    {
        for (; _First1 != _Last1; ++_First1, ++_Result)
        {
            auto _Value = *_First1;
            auto _InnerFirst = _First2;
            auto _ResultFirst = begin(*_Result);

            for (; _InnerFirst != _Last2; ++_InnerFirst, ++_ResultFirst)
            {
                *_ResultFirst += *_InnerFirst * _Value;
            }
        }
    }

    using namespace std;
    template<typename _Type, typename _EngineType = default_random_engine>
    class RecurrentNeuralNetwork
    {
    public:
        using size_type = size_t;
        using _ActivationFunction = _Type(*)(_Type);

        RecurrentNeuralNetwork() = default;
        ~RecurrentNeuralNetwork() = default;

        template<typename... _Args>
        RecurrentNeuralNetwork(
            size_type _TimeCount,
            size_type _Dimension,
            size_type _LayerCount,
            _Type __learningrate,
            _Args&&... __args
        ) :
            _Hidden(_TimeCount, VectorType<_Type>(_LayerCount)),
            _Output(_TimeCount, VectorType<_Type>(_Dimension)),
            _U(_LayerCount, VectorType<_Type>(_Dimension)),
            _V(_Dimension, VectorType<_Type>(_LayerCount)),
            _W(_LayerCount, VectorType<_Type>(_LayerCount)),
            _DeltaHidden(_LayerCount),
            _DeltaOutput(_Dimension),
            _DeltaU(_LayerCount, VectorType<_Type>(_Dimension)),
            _DeltaV(_Dimension, VectorType<_Type>(_LayerCount)),
            _DeltaW(_LayerCount, VectorType<_Type>(_LayerCount)),
            _LearningRate(__learningrate),
            _Engine(forward<_Args>(__args)...)
        {
            Reset();
        }

        void Reset(void)
        {
            auto&& _Range = (_Type(1) / sqrt(_Hidden.size()));
            uniform_real_distribution<_Type> _Distribution(-_Range, _Range);

            auto ResetWight = [&](auto&& _WightVector) { for (auto&& _Vector : _WightVector)
            {
                std::generate(std::begin(_Vector), std::end(_Vector), std::bind(_Distribution, std::ref(_Engine)));
            }; };

            ResetWight(_U);
            ResetWight(_V);
            ResetWight(_W);
            //for (auto&& _Vector : _U)
            //{
            //    std::generate(std::begin(_Vector), std::end(_Vector), std::bind(_Distribution, std::ref(_Engine)));
            //}

            //for (auto&& _Vector : _V)
            //{
            //    std::generate(std::begin(_Vector), std::end(_Vector), std::bind(_Distribution, std::ref(_Engine)));
            //}

            //for (auto&& _Vector : _W)
            //{
            //    std::generate(std::begin(_Vector), std::end(_Vector), std::bind(_Distribution, std::ref(_Engine)));
            //}
        }

        template<typename _InputType>
        void ForwardPropagation(_InputType&& _Input)
        {
            auto&& _TimeCount = _Hidden.size();
            for (decltype(_TimeCount) _Time(0); _Time < _TimeCount; ++_Time)
            {
                // First Time
                if (decltype(_TimeCount)(0) == _Time)
                {
                    transform(begin(_U), end(_U), begin(_Hidden[0]),
                        [
                            _InputFirst = std::begin(_Input[0]),
                            _InputLast = std::end(_Input[0])
                        ](auto&& _Vector)
                    {
                        auto&& _Sum = inner_product(_InputFirst, _InputLast, begin(_Vector), _Type(0));
                        return tanh(_Sum);
                    });

                    //cout << "_Input[0]" << endl << _Input[0] << endl << endl;
                    //cout << "_U" << endl << _U << endl << endl;
                    //cout << "_Hidden[0]" << endl << _Hidden[0] << endl << endl;
                }
                else
                {
                    transform(begin(_U), end(_U), begin(_W), begin(_Hidden[_Time]),
                        [
                            _InputFirst = std::begin(_Input[_Time]),
                            _InputLast = std::end(_Input[_Time]),
                            _PrevHiddenFirst = std::begin(_Hidden[_Time - 1]),
                            _PrevHiddenLast = std::end(_Hidden[_Time - 1])
                        ] (auto&& _Vector1, auto&& _Vector2)
                    {
                        auto&& _Sum1 = inner_product(_InputFirst, _InputLast, begin(_Vector1), _Type(0));
                        auto&& _Sum2 = inner_product(_PrevHiddenFirst, _PrevHiddenLast, begin(_Vector2), _Type(0));

                        return tanh(_Sum1 + _Sum2);
                    });

                    //cout << "_Input[0]" << endl << _Input[0] << endl << endl;
                    //cout << "_U" << endl << _U << endl << endl;
                    //cout << "_Hidden[0]" << endl << _Hidden[0] << endl << endl;
                }

                transform(begin(_V), end(_V), begin(_Output[_Time]),
                    [
                        _HiddenFirst = begin(_Hidden[_Time]),
                        _HiddenLast = end(_Hidden[_Time])
                    ](auto&& _Vector)
                {
                    return inner_product(_HiddenFirst, _HiddenLast, begin(_Vector), _Type(0));
                });

                Softmax(begin(_Output[_Time]), end(_Output[_Time]));

                //cout << "_Hidden[_Time]" << endl << _Hidden[_Time] << endl << endl;
                //cout << "_V" << endl << _V << endl << endl;
                //cout << "_Output[_Time]" << endl << _Output[_Time] << endl << endl;
            }
        }

        template<typename _InputType, typename _TargetType>
        void BackPropagationThroughTime(_InputType&& _Input, _TargetType&& _Target)
        {
            for_each(begin(_DeltaU), end(_DeltaU), [](auto&& _Vector)
            { fill(begin(_Vector), end(_Vector), _Type(0)); });

            for_each(begin(_DeltaV), end(_DeltaV), [](auto&& _Vector)
            { fill(begin(_Vector), end(_Vector), _Type(0)); });

            for_each(begin(_DeltaW), end(_DeltaW), [](auto&& _Vector)
            { fill(begin(_Vector), end(_Vector), _Type(0)); });

            for (auto&& _Time = _Hidden.size(); _Time-- != 0;)
            {
                transform(begin(_Target[_Time]), end(_Target[_Time]), begin(_Output[_Time]), begin(_DeltaOutput),
                    [
                        _Sigma = accumulate(begin(_Target[_Time]), end(_Target[_Time]), _Type(0))
                    ](auto&& _TargetValue, auto&& _OutputValue)
                {
                    return _Sigma * _OutputValue - _TargetValue;
                });

                auto&& _HiddenTime = _Hidden[_Time];
                OuterProduct(begin(_DeltaOutput), end(_DeltaOutput),
                    begin(_HiddenTime), end(_HiddenTime), begin(_DeltaV));

#define TestOtherCode
#ifndef TestOtherCode
                fill(begin(_DeltaHidden), end(_DeltaHidden), _Type(0));
                ForEach(begin(_V), end(_V), begin(_DeltaOutput), [&_DeltaHidden](auto&& _Vector, auto&& _Delta)
                {
                    ForEach(begin(_DeltaHidden), end(_DeltaHidden), begin(_Vector),
                        [_Delta](auto&& _Sum, auto&& _Weight)
                    {
                        _Sum += _Delta * _Weight;
                    })
                });
                ForEach(begin(_DeltaHidden), end(_DeltaHidden), begin(_Hidden[_Time]),
                    [](auto&& _Sum, auto&& _Value)
                {
                    _Sum *= _Type(1) - _Value * _Value;
                })
#else
                auto&& _LayerCount = _DeltaHidden.size();
                auto&& _DimensionCount = _DeltaOutput.size();
                for (decltype(_LayerCount) _Layer(0); _Layer < _LayerCount; ++_Layer)
                {
                    _Type _Sum(0);
                    for (decltype(_DimensionCount) _Dimension(0); _Dimension < _DimensionCount; ++_Dimension)
                    {
                        _Sum += _V[_Dimension][_Layer] * _DeltaOutput[_Dimension];
                    }

                    if (_Hidden.size() - 1 != _Time)
                    {
                        auto&& _WLayer = _W[_Layer];
                        auto&& _NextHiddenTime = _Hidden[_Time + 1];
                        for (decltype(_LayerCount) _InnerLayer(0); _InnerLayer < _LayerCount; ++_InnerLayer)
                        {
                            _Sum += _WLayer[_InnerLayer] * _NextHiddenTime[_InnerLayer];
                        }
                    }

                    _DeltaHidden[_Layer] = _Sum * (_Type(1) - _HiddenTime[_Layer] * _HiddenTime[_Layer]);
                }
#endif

                if (0 != _Time)
                {
                    OuterProduct(begin(_DeltaHidden), end(_DeltaHidden),
                        begin(_Hidden[_Time - 1]), end(_Hidden[_Time - 1]), begin(_DeltaW));
                }

                OuterProduct(begin(_DeltaHidden), end(_DeltaHidden),
                    begin(_Input[_Time]), end(_Input[_Time]), begin(_DeltaU));
            }

            ForEach(begin(_U), end(_U), begin(_DeltaU), [this](auto&& _Vector1, auto&& _Vector2)
            {
                ForEach(begin(_Vector1), end(_Vector1), begin(_Vector2), [&](auto&& _Value1, auto&& _Value2)
                {
                    _Value1 += _Value2 * _LearningRate;
                });
            });
            ForEach(begin(_V), end(_V), begin(_DeltaV), [this](auto&& _Vector1, auto&& _Vector2)
            {
                ForEach(begin(_Vector1), end(_Vector1), begin(_Vector2), [&](auto&& _Value1, auto&& _Value2)
                {
                    _Value1 += _Value2 * _LearningRate;
                });
            });
            ForEach(begin(_W), end(_W), begin(_DeltaW), [this](auto&& _Vector1, auto&& _Vector2)
            {
                ForEach(begin(_Vector1), end(_Vector1), begin(_Vector2), [&](auto&& _Value1, auto&& _Value2)
                {
                    _Value1 += _Value2 * _LearningRate;
                });
            });
        }

        const auto& GetOutput(void)
        {
            return _Output;
        }

    private:
        VectorType<VectorType<_Type>> _Hidden;
        VectorType<VectorType<_Type>> _Output;
        VectorType<VectorType<_Type>> _U;
        VectorType<VectorType<_Type>> _V;
        VectorType<VectorType<_Type>> _W;

        VectorType<_Type> _DeltaHidden;
        VectorType<_Type> _DeltaOutput;
        VectorType<VectorType<_Type>> _DeltaU;
        VectorType<VectorType<_Type>> _DeltaV;
        VectorType<VectorType<_Type>> _DeltaW;

        _Type _LearningRate;

        _EngineType _Engine;
    };
}