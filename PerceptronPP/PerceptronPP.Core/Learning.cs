using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronPP.Core
{
    public class Learning
    {
        public static void Learn(Network network, int batchSize, double learningCoefficient, double[][] trainData, double[][] expectedOutputs)
        {
            double[] output;
            for (int i = 0; i < trainData.Length; i++)
            {
                Iterate(network, trainData[i], expectedOutputs[i]);

                if ((i + 1) % batchSize == 0)
                {
                    //Thread.Sleep();
                    network.GradientDescend(learningCoefficient);
                    Console.Clear();

                    Console.WriteLine(network.GetCost());
                    Console.WriteLine(i);
                    Thread.Sleep(10);
                    network.ResetCost();
                }
            }
            if (trainData.Length % batchSize == 0)
                try { network.GradientDescend(learningCoefficient); }
                catch { }
        }

        public static void Iterate(Network network, double[] input, double[] expectedOutput)
        {
            var output = network.Compute(input);
            //Console.WriteLine(String.Join(" ,",output));
            //Console.WriteLine(String.Join(" ,", expectedOutput));

            network.BackPropagate(output, expectedOutput);
            network.CalculateCost(output, expectedOutput);
        }

        public static double[] IntToExpectedOutputArray(int value)
        {
            return Enumerable.Range(0, 10).Select((e, i) => i == value ? 1.0 : 0.0).ToArray();
        }
    }
}
