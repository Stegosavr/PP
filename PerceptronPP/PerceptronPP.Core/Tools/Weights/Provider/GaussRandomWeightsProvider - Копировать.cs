namespace PerceptronPP.Core.Tools.Weights.Provider;

public class GaussianRandomWeightsProvider : IWeightsProvider
{
	private readonly double _distribution;

	public GaussianRandomWeightsProvider(int layer, double distribution)
	{
		_distribution = distribution;
	}
	
	public double GetWeight(int _, int __)
	{
		Random rand = new Random(); //reuse this if you are generating many
		double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
		double u2 = 1.0 - rand.NextDouble();
		double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
					 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
		double randNormal =
					 0 + _distribution * randStdNormal; //random normal(mean,stdDev^2)
		return randNormal;
	}
}