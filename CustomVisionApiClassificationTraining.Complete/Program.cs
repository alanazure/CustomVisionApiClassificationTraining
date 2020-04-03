using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace CustomVisionApiClassificationTraining.Complete
{
    class Program
    {
        // Set your endpoint and key here
        static string Endpoint = "";
        static string TrainingKey = "";

        // Folder for training images
        static string TrainingImageFolder = @"C:\AIData\Simpsons\simpsons_dataset";

        static string ProjectName = "Test01";

        static CustomVisionTrainingClient TrainingClient;

        static async Task Main(string[] args)
        {
            Console.WriteLine("Custom Vision API - Classification - Training");

            using (var classificationTrainer = 
                new CustomVisionClassificationTrainer(Endpoint, TrainingKey))
            {
                classificationTrainer.ImagesPerClass = 10;
                classificationTrainer.NumberOfClasses = 3;

                var projectId = await classificationTrainer.CreateProjectAsync(ProjectName);

                await classificationTrainer.UploadTrainingImages(projectId, TrainingImageFolder);

                var iteration = await classificationTrainer.TrainModel(projectId);

                Console.WriteLine("Press enter to display statistics.");
                Console.ReadLine();

                await classificationTrainer.DisplayStatistics(projectId, iteration);

            }





        }



    }
}
