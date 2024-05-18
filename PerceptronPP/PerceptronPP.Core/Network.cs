using MathNet.Numerics.LinearAlgebra;
using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP.Core;

public class Network
{
	private readonly Layer[] _layers;
	private readonly IComputable _activationComputable;
	private readonly int[] _neuronCounts;

	private int _iterations;
	private double _cost;

	public Network(IComputable computable, params int[] neuronCounts)
	{
		_activationComputable = computable;
		_neuronCounts = neuronCounts;
		_layers = neuronCounts
			.Where((_, i) => i < neuronCounts.Length - 1)
			.Select((e, i) => new Layer(e, neuronCounts[i + 1]))
			.Append(new Layer(neuronCounts[^1]))
			.ToArray();
	}

	public int GetNeuronCount(int layer)
	{
		return _neuronCounts[layer];
	}

	public int GetNeuronCount()
	{
		return _neuronCounts.Sum();
	}

	public Network SetWeights(IWeightsFactory provider)
	{
		for (var i = 0; i < _layers.Length-1; i++)
			_layers[i].SetWeights(provider.Create(i));

		return this;
	}

	public Network SetBiases(IBiasProvidable biasProvidable)
	{
		for (var i = 0; i < _layers.Length-1; i++)
		{
			_layers[i].SetBiases(Enumerable
				.Range(0, _neuronCounts[i+1])
				.Select(n => biasProvidable.GetBias(i, n))
				.ToArray()
			);
		}

		return this;
	}

	public double[] Compute(params double[] input)
	{
		var matrix = ComputeMatrix(input);
		//matrix = _layers.SkipLast(1).Last().ComputeLastOutput(matrix);//////

		var result = new double[matrix.ColumnCount];
		for (var i = 0; i < matrix.ColumnCount; i++)
		{
			result[i] = matrix[0, i];
		}

		return result;
	}

	public Matrix<double> ComputeMatrix(double[] input)
	{
		return _layers.SkipLast(1).Aggregate
		(
			Layer.MatrixArray(input),
			(current, layer) => layer.ComputeOutput(_activationComputable, current)
		);
	}

	public void CalculateCost(double[] output, double[] expectedOutput)
	{
		for (var i =0; i < output.Length; i++)
        {
			_cost += Math.Pow(output[i]-expectedOutput[i],2);
        }
	}

	public double GetCost()
    {
		return _cost;
    }

	public void ResetCost()
    {
		_cost = 0;
    }

	public void BackPropagate(double[] networkOutput,double[] expectedNetworkOutput)
    {
		var output = 2 * (Layer.MatrixArray(networkOutput) - Layer.MatrixArray(expectedNetworkOutput));
		_layers.Reverse().Skip(1).Aggregate
		(
			output,
			(current, layer) => layer.BackPropagate(current)
		);

		_iterations++;
	}

	public void GradientDescend(double coefficient)
    {
		if (_iterations == 0) throw new InvalidOperationException("Back propagation needs to be called first");

		foreach (var layer in _layers.SkipLast(1))
			layer.GradientDescend(coefficient, _iterations);

		_iterations = 0;
	}

	public Layer GetLayer(int layer)
	{
		return _layers[layer];
	}
}