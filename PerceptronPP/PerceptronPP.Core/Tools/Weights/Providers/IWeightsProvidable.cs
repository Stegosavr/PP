using PerceptronPP.Core.Tools.Weights;

namespace PerceptronPP.Core.Tools.Providers;

public interface IWeightsProvidable
{
	public IWeightsAccessible GetAccessor(int layer);
}