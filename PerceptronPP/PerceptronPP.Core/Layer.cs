﻿using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PerceptronPP.Core.Errors;

namespace PerceptronPP.Core;

public class Layer
{
	private readonly int _neuronsCount;
	private readonly Matrix<double> _weights;
	private readonly Matrix<double> _biases;

	public Layer(int neurons, int nextNeurons = 0)
	{
		_neuronsCount = neurons;
		_weights = CreateMatrix.Dense<double>(_neuronsCount, nextNeurons);
		_biases = CreateMatrix.Dense<double>(1, nextNeurons);
	}

	public void RandomizeWeights(Func<int, double> randomizer)
	{
		for (var i = 0; i < _weights.RowCount; i++)
		for (var j = 0; j < _weights.ColumnCount; j++)
		{
			_weights[i, j] = randomizer(_neuronsCount);
		}
	}

	public Matrix<double> ComputeOutput(IComputable computable, Matrix<double> input)
	{
		if (input.ColumnCount != _neuronsCount) throw new IncorrectNeuronCountException();
		var output = input * _weights + _biases;

		for (var i = 0; i < output.ColumnCount; i++)
			output[0, i] = computable.Compute(output[0, i]);

		return output;
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