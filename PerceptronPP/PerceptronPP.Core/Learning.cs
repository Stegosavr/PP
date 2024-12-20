﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PerceptronPP.Core.Tools.GradientDescent.Optimizers;
using System.IO;
using OfficeOpenXml;
using OfficeOpenXml.Style;
using PerceptronPP.Core.FileManager.Excel;

namespace PerceptronPP.Core;

public class Learning
{
    public static void LoopLearn
        (Network initialNetwork,IOptimizer otimizer,
        int batchSize,double learningCoefficient, 
        double[][] trainData, int[] expectedOutputs,
        Parameters parameterToChange, double changeTo,
        int iterationsNumber)
    {
        
        for (int iteration = 0; iteration < iterationsNumber; iteration++)
        {

        }
        ExcelWriter.SaveXLSX(@"..\..\..\..\LearningResults\parameters.xlsx", new ExcelLearningData[]
        {
            new ExcelLearningData{Optimizer = "SDSD",Time = "10"},
            new ExcelLearningData{Optimizer = "SDSD",Time = "10",BatchSize ="10",LearningCoefficient="6"}
        });
    }

    public static void Learn(Network network, IOptimizer optimizer, int batchSize, double learningCoefficient, double[][] trainData, int[] expectedOutputs)
    {
        //Console.
        var coef = learningCoefficient;
        var delta = (learningCoefficient - 0.05) / trainData.Length;
        double[] output;
        for (int i = 0; i < trainData.Length; i++)
        {
            Iterate(network, trainData[i],  expectedOutputs[i]);
            //coef -= delta;
            //if ((i + 1) % 15000 == 0 && batchSize != 1)
            //    batchSize-=1;
            if ((i + 1) % batchSize == 0)
            {
                //Thread.Sleep();
                network.GradientDescent(optimizer,coef);
                if ((i + 1) % (batchSize * 100) == 0)
                {
                    WriteToConsole(network.GetCost().ToString(), i.ToString());
                }
                network.ResetCost();
            }
        }
        if (trainData.Length % batchSize == 0)
            try { network.GradientDescent(optimizer,learningCoefficient); }
            catch { }
    }

    public static double Test(Network network, double[][] testData, int[] expectedOutputs)
    {
        Console.Clear();
        int count = 0;
        int correct = 0;
        double[] output;
        for (int i = 0; i < testData.Length; i++)
        {
            output = Iterate(network, testData[i], expectedOutputs[i]);
            if (Check(output, expectedOutputs[i]))
                correct++;
            count++;
            
            if ((i + 1)%100 == 0)
                WriteToConsole((correct / (double)count * 100).ToString(), i.ToString());

        }
        Console.SetCursorPosition(0, 4);

        return correct / (double)count * 100;
    }

    private static bool Check(double[] output, int expectedOutput)
    {
        var label = expectedOutput;
        var outputLabel = output.Select((e, i) => Tuple.Create(e, i)).OrderByDescending(tuple => tuple.Item1).First().Item2;
        if (label == outputLabel) return true;
        return false;
    }

    public static double[] Iterate(Network network, double[] input, int expectedOutput)
    {
        var output = network.Compute(input);
        //Console.WriteLine(String.Join(" ,",output));
        //Console.WriteLine(String.Join(" ,", expectedOutput));
        network.BackPropagate(output,Learning.IntToExpectedOutputArray(expectedOutput));
        network.CalculateCost(output, Learning.IntToExpectedOutputArray(expectedOutput));
        
        return output;
    }

    public static double[] IntToExpectedOutputArray(int value)
    {
        return Enumerable.Range(0, 10).Select((e, i) => i == value ? 1.0 : 0.0).ToArray();
    }

    public static void WriteToConsole(string s,string iteration)
    {
        Console.WriteLine(s);
        Console.WriteLine(iteration);
        var (_, top) = Console.GetCursorPosition();
        Console.SetCursorPosition(0, top - 2);
    }
}
