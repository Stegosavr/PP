using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PerceptronPP.Core.Exceptions;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights;
using PerceptronPP.Core.Tools.Weights.Provider;

namespace PerceptronPP.Core;

public class Layer
{
	private readonly int _neuronsCount;
	private Matrix<double> _weights;//deReadonlyized
	private Matrix<double> _biases;//deReadonlyized

	private Matrix<double> _input;
	private readonly BackPropagationData _backPropData;

	public Layer(int neurons, int nextNeurons = 0)
	{
		_neuronsCount = neurons;
		_weights = CreateMatrix.Dense<double>(_neuronsCount, nextNeurons);
		_biases = CreateMatrix.Dense<double>(1, nextNeurons);

		_input = CreateMatrix.Dense<double>(1, _neuronsCount);
		_backPropData = new BackPropagationData(neurons,nextNeurons);
	}

	public void SetWeights(IWeightsProvider weights)
	{
		for (var i = 0; i < _weights.RowCount; i++)
		for (var j = 0; j < _weights.ColumnCount; j++)
		{
			_weights[i, j] = weights.GetWeight(i, j);
		}
	}

	public void SetBiases(params double[] biases)
	{
		for (var i = 0; i < _biases.ColumnCount; i++)
			_biases[0, i] = biases[i];
	}
	
	public Matrix<double> ComputeOutput(IComputable computable, Matrix<double> input)
	{
		if (input.ColumnCount != _neuronsCount) throw new IncorrectNeuronCountException();
		var output = input * _weights + _biases;

		for (var i = 0; i < output.ColumnCount; i++)
		{
			_backPropData.NeuronsInputSignalDerivative[0, i] = computable.ComputeDerivative(output[0, i]);
			output[0, i] = computable.Compute(output[0, i]);
		}

		//input.CopyTo(_input);
		_input = input;

		return output;
	}

	public Matrix<double> ComputeLastOutput(Matrix<double> input)
	{
		if (input.ColumnCount != _neuronsCount) throw new IncorrectNeuronCountException();
		var output = input * _weights + _biases;

		
		output = SoftmaxComputable.Compute(output);
		_backPropData.NeuronsInputSignalDerivative = SoftmaxComputable.ComputeDerivative(output);


		//input.CopyTo(_input);
		_input = input;

		return output;
	}

	public Matrix<double> BackPropagate(Matrix<double> output)
	{
		var (weightsDer, biasesDer, activationsDer, neuronInputDer) = 
			(_backPropData.WeightsDerivative, _backPropData.BiasesDerivative, 
			_backPropData.ActivationDerivative,_backPropData.NeuronsInputSignalDerivative);
		var outputByinputDer = CreateMatrix.Dense<double>(1,output.ColumnCount);
		for (var j = 0; j < _biases.ColumnCount; j++)
		{
			outputByinputDer[0, j] += output[0, j] * neuronInputDer[0, j];
		}


		for (var i = 0; i < _weights.RowCount; i++)
        {
			for (var j = 0; j < _weights.ColumnCount; j++)
				weightsDer[i, j] += outputByinputDer[0, j] * _input[0, i];
        }
		for (var j = 0; j < _biases.ColumnCount; j++)
        {
			biasesDer[0, j] += outputByinputDer[0, j];
		}
		for (var i = 0; i < _weights.RowCount; i++)
		{
			var activationsDerSum = 0.0;
			for (var j = 0; j < _weights.ColumnCount; j++)
				activationsDerSum += outputByinputDer[0, j] * _weights[i, j];
			activationsDer[0,i] += activationsDerSum / _weights.ColumnCount;
		}
		return activationsDer;
	}

	public void GradientDescent(double coefficient, int iterations)
    {
		_weights -= _backPropData.WeightsDerivative / iterations * coefficient;
		_biases -= _backPropData.BiasesDerivative / iterations  * coefficient;

		_backPropData.Clear();
	}

	public static Matrix<double> MatrixArray(double[] input)
	{
		var matrixArray = new double[1, input.Length];
		for (var i = 0; i < input.Length; i++)
		{
			matrixArray[0, i] = input[i];
		}

		return DenseMatrix.OfArray(matrixArray);
	}
}