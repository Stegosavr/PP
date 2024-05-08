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
	private readonly Matrix<double> _weights;
	private readonly Matrix<double> _biases;

	private readonly Matrix<double> _input;
	private readonly Matrix<double> _neuronsInputSignalDerivative;

	public Layer(int neurons, int nextNeurons = 0)
	{
		_neuronsCount = neurons;
		_weights = CreateMatrix.Dense<double>(_neuronsCount, nextNeurons);
		_biases = CreateMatrix.Dense<double>(1, nextNeurons);

		_input = CreateMatrix.Dense<double>(1, _neuronsCount);
		_neuronsInputSignalDerivative = CreateMatrix.Dense<double>(1, nextNeurons);
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
			output[0, i] = computable.Compute(output[0, i]);
			_neuronsInputSignalDerivative[0, i] = computable.ComputeDerivative(output[0, i]);
		}

		input.CopyTo(_input);

		return output;
	}

	public Matrix<double> BackPropagate(Matrix<double> output)
	{
		var weightsDerivative = CreateMatrix.Dense<double>(_weights.RowCount,_weights.ColumnCount);
		var biasesDerivative = CreateMatrix.Dense<double>(1,_biases.ColumnCount);
		var activationDerivative = CreateMatrix.Dense<double>(1,_neuronsCount);

		for (var i = 0; i < _weights.RowCount; i++)
        {
			for (var j = 0; j < _weights.ColumnCount; j++)
				weightsDerivative[i, j] = output[0,j] * _neuronsInputSignalDerivative[0, j] * _input[0, i];
        }
		for (var j = 0; j < _biases.RowCount; j++)
        {
			biasesDerivative[0, j] = output[0, j] * _neuronsInputSignalDerivative[0, j];
		}
		for (var i = 0; i < _weights.RowCount; i++)
		{
			for (var j = 0; j < _weights.ColumnCount; j++)
				activationDerivative[0,i] += output[0, j] * _neuronsInputSignalDerivative[0, j] * _weights[i, j];
			activationDerivative[0,i] /= _weights.ColumnCount;
		}

		return activationDerivative;
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