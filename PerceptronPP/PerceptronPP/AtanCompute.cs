using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronPP;

internal class AtanCompute : IComputable
{
	public static readonly AtanCompute Share = new();

	public double Compute(double x)
	{
		return Math.Atan(x) / Math.PI + 0.5;
	}

	public double ComputeDerivative(double x)
	{
		return 1 / (x * x + 1) / Math.PI;
	}
}