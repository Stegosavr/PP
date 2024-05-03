using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using PerceptronPP.Core.Errors;

namespace PerceptronPP.Core;

public class Network
{
	// public readonly Perceptron Perceptron;
	private readonly Layer[] _layers;
	private readonly IComputable _activationComputable;

	public Network(IComputable computable, params int[] neuronCounts)
	{
		_activationComputable = computable;
		_layers = neuronCounts
			.Where((_, i) => i < neuronCounts.Length - 1)
			.Select((e, i) => new Layer(e, neuronCounts[i + 1]))
			.Append(new Layer(neuronCounts[^1]))
			.ToArray();
	}

	public Network RandomizeWeights(Func<int, double> randomizer)
	{
		foreach (var layer in _layers)
			layer.RandomizeWeights(randomizer);
		return this;
	}

	public Network RandomizeWeights()
	{
		return RandomizeWeights(count =>
		{
			var distribution = (1 / Math.Sqrt(count));
			return new Random().NextDouble() * distribution * 2 - distribution;
		});
	}

	public double[] Compute(params double[] input)
	{
		var matrix = _layers.SkipLast(1).Aggregate
		(
			Layer.MatrixArray(input),
			(current, layer) => layer.ComputeOutput(_activationComputable, current)
		);
		var result = new double[matrix.ColumnCount];
		for (var i = 0; i < matrix.ColumnCount; i++)
		{
			result[i] = matrix[0, i];
		}

		return result;
	}
}