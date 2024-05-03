using PerceptronPP.Core.Tools.Weights.Provider;

namespace PerceptronPP.Core.Tools.Weights.Factory;

public class RandomWeightsFactory : IWeightsFactory
{
	private readonly Func<int, int> _getCount;

	public RandomWeightsFactory(Func<int, int> getCount)
	{
		_getCount = getCount;
	}
	public IWeightsProvider Create(int layer)
	{
		return new RandomWeightsProvider(_getCount(layer));
	}
}