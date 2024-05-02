using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace PerceptronPP
{
    internal class Perceptron
    {
        public Func<double, double> ActivationFunction = AtanCompute.Share.Compute;
        int LayersCount;
        public Layer[] Layers;
        int ForwardIterationsCount = 0;

        public Perceptron(params Layer[] layers)
        {
            LayersCount = layers.Length;
            Layers = layers;
            foreach (Layer layer in layers)
                layer.Network = this;
        }

        public void Randomize()
        {
            for (int i = 0; i < LayersCount - 1; i++)
            {
                Layers[i].RandomizeWeights();
            }
        }

        public void ForwardPropagation()
        {
            for (int i = 0; i < LayersCount-1; i++)
            {
                Layers[i].ForwardPropagation(Layers[i+1]);
            }
        }

        public Matrix<double> GetOutput()
        {
            return Layers.Last().Input;
        }
    }
}
