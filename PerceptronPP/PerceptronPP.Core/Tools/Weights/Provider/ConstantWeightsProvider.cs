namespace PerceptronPP.Core.Tools.Weights.Provider;

public class ConstantWeightsProvider : IWeightsProvider
{
	private readonly double[,] _weights;

	public ConstantWeightsProvider(double[,] weights)
	{
		_weights = weights;
	}

	public double GetWeight(int current, int next)
	{
		return _weights[current, next];
	}
}