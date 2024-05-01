using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace PerceptronPP
{
    internal class Layer
    {
        int NeuronsCount;
        int NextLayerNeuronsCount;
        public Matrix<double> Input;
        Matrix<double> Output;
        public Matrix<double> Weights;
        Matrix<double> Biases;
        public Perceptron Network;

        public Layer(int neurons,int nextNeurons = 0)
        {
            NeuronsCount = neurons;
            NextLayerNeuronsCount = nextNeurons;
            //Network = network;
            SetWeights();
            SetBias();
        }

        public void RandomizeWeights()
        {
            var randomizer = () =>
            {
                var distribution = (1 / Math.Sqrt(NeuronsCount));
                return new Random().NextDouble() * distribution * 2 - distribution;
            };
            
            for (int i = 0; i < Weights.RowCount; i++)
                for (int j = 0; j < Weights.ColumnCount; j++)
                {
                    Weights[i,j] = randomizer();
                }
        }

        public void SetWeights()
        {
            Weights = CreateMatrix.Dense<double>(NeuronsCount, NextLayerNeuronsCount);
        }

        public void SetBias()
        {
            Biases = CreateMatrix.Dense<double>(1,NextLayerNeuronsCount);
        }

        public void ForwardPropagation(Layer nextLayer)
        {
            Output = Input * Weights + Biases;

            for (int i = 0; i < Output.ColumnCount; i++)
                Output[0, i] = Network.ActivationFunction(Output[0,i]); 

            nextLayer.Input = Output;
        }
    }
}
