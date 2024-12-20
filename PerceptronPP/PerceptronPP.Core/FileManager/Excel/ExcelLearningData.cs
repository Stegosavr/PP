﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptronPP.Core.FileManager.Excel;

public class ExcelLearningData
{
    public string HiddenLayers;
    public string ActivationFunction;
    public string Optimizer;

    public string BatchSize;
    public string BatchSize2;

    public string LearningCoefficient;
    public string LearningCoefficient2;

    public string WeightsDistributionType;
    public string WeightsDistribution;

    public string TrainingDataSize;
    public string PredictionEfficiency;
    public string Time;


    public ExcelLearningData()
    {
    }
}
