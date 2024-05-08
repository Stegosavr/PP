using PerceptronPP.Core.Tools.Biases;
using PerceptronPP.Core.Tools.Computable;
using PerceptronPP.Core.Tools.Weights.Factory;

namespace PerceptronPP.Core;

public class Network
{
	private readonly Layer[] _layers;
	private readonly IComputable _activationComputable;
	private readonly int[] _neuronCounts;

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

	public double CalculateCost(double[] output, double[] expectedOutput)
	{
		double cost = 0;
		for (int i =0; i < output.Length; i++)
        {
			cost += Math.Pow(output[i]-expectedOutput[i],2);
        }
		return cost;
	}

	public void BackPropagate(double[] networkOutput,double[] expectedNetworkOutput)
    {
		var output = 2 * (Layer.MatrixArray(networkOutput) - Layer.MatrixArray(expectedNetworkOutput));
		_layers.Reverse().Skip(1).Aggregate
		(
			output,
			(current, layer) => layer.BackPropagate(current)
		);
	}

	public void GradientDescend(double coefficient)
    {
		foreach (var layer in _layers.SkipLast(1))
			layer.GradientDescend(coefficient);
    }
}