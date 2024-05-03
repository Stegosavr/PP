namespace PerceptronPP.Core.Tools.Weights;

public interface IWeightsAccessible
{
	public double GetWeight(int current, int next);
}