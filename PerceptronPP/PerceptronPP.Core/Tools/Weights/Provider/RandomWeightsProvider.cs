namespace PerceptronPP.Core.Tools.Weights.Provider;

public class RandomWeightsProvider : IWeightsProvider
{
	private readonly double _distribution;

	public RandomWeightsProvider(int layer)
	{
		//_distribution = (1 / Math.Sqrt(layer));

		_distribution = 1;
	}
	
	public double GetWeight(int _, int __)
	{
		//return new Random().NextDouble() * _distribution * 2 - _distribution *1;

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