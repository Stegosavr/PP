namespace PerceptronPP.Core.Tools.Biases;

public class BiasMapProvider : IBiasProvidable
{
	private double[,] _biases;

	public BiasMapProvider(double[,] biases)
	{
		_biases = biases;
	}

	public double GetBias(int layer, int neuron)
	{
		return _biases[layer, neuron];
	}
}