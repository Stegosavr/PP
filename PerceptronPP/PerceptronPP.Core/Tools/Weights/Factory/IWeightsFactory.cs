using PerceptronPP.Core.Tools.Weights.Provider;

namespace PerceptronPP.Core.Tools.Weights.Factory;

public interface IWeightsFactory
{
	public IWeightsProvider Create(int layer);
}