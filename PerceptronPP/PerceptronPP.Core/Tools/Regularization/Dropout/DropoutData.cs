using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronPP.Core.Tools;

public class DropoutData
{
    public readonly double Probability;
    public readonly bool Testing;

    public DropoutData(double probability, bool testing)
    {
        Probability = probability;
        Testing = testing;
    }
}
