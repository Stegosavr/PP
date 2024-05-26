namespace PerceptronPP.Core.Tools.Weights.Provider;

public class MapWightsProvider : IWeightsProvider
{
	private readonly double[,] _weights;

	public MapWightsProvider(double[,] weights)
	{
		_weights = weights;
	}

	public double GetWeight(int current, int next)
	{
		return _weights[current, next];
	}
}