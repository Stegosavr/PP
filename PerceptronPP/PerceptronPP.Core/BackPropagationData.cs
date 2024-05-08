using MathNet.Numerics.LinearAlgebra;

namespace PerceptronPP.Core;

public class BackPropagationData
{
    public readonly Matrix<double> NeuronsInputSignalDerivative;

    public readonly Matrix<double> WeightsDerivative;
    public readonly Matrix<double> BiasesDerivative;
    public readonly Matrix<double> ActivationDerivative;

    public BackPropagationData(int neurons, int nextNeurons)
    {
        NeuronsInputSignalDerivative = CreateMatrix.Dense<double>(1, nextNeurons);
    
        WeightsDerivative = CreateMatrix.Dense<double>(neurons, nextNeurons);
        BiasesDerivative = CreateMatrix.Dense<double>(1, nextNeurons);
        ActivationDerivative = CreateMatrix.Dense<double>(1, neurons);
    }

    public void Clear()
    {
        NeuronsInputSignalDerivative.Clear();
        WeightsDerivative.Clear();
        BiasesDerivative.Clear();
        ActivationDerivative.Clear();
    }
}
