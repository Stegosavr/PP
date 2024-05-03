using PerceptronPP.Core.Tools.Weights;

namespace PerceptronPP.Core.Tools.Providers;

public class RandomWeightsProvider : IWeightsProvidable
{
	private readonly Func<int, int> _getCount;

	public RandomWeightsProvider(Func<int, int> getCount)
	{
		_getCount = getCount;
	}
	public IWeightsAccessible GetAccessor(int layer)
	{
		return new RandomWeightsAccessor(_getCount(layer));
	}
}