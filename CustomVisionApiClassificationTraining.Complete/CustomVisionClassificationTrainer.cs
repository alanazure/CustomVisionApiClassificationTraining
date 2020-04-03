using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace CustomVisionApiClassificationTraining.Complete
{
    class CustomVisionClassificationTrainer : IDisposable
    {

        private static CustomVisionTrainingClient m_TrainingClient;

        public int ImagesPerClass { get; set; }
        public int NumberOfClasses { get; set; }

        public CustomVisionClassificationTrainer(string endpoint, string trainingKey)
        {
            // Create a training client
            m_TrainingClient = new 
                CustomVisionTrainingClient { Endpoint = endpoint, ApiKey = trainingKey };

            ImagesPerClass = 50;
            NumberOfClasses = int.MaxValue;
        }



        public async Task<Guid> CreateProjectAsync(string name)
        {
            // Create a project and return the Id
            Console.WriteLine($"Creating project { name }...");
            var project = await m_TrainingClient.CreateProjectAsync(name, classificationType: "Multiclass");
            Console.WriteLine($"Done! ProjectId: { project.Id }");
            Console.WriteLine();
            return project.Id;
        }


        public async Task UploadTrainingImages(Guid projectId, string trainingImageFolder)
        {
            Console.WriteLine("Uploading training images...");

            // Get a list of the folders that have the minimum number of required images.
            var trainingFolderNames = 
                GetFolderNames(trainingImageFolder, ImagesPerClass, NumberOfClasses);
            

            // Iterate through the training folders and upload the images
            foreach (var trainingFolderName in trainingFolderNames)
            {
                Console.WriteLine($"    Uploading and tagging { trainingFolderName }...");
                var trainingFolder = Path.Combine(trainingImageFolder, trainingFolderName);
                await TagTestImages(projectId, trainingFolder, trainingFolderName);
            }
            Console.WriteLine("Done!");
            Console.WriteLine();
        }

        public async Task<Iteration> TrainModel(Guid projectId)
        {
            // Train the project with the default settings
            var iteration = await m_TrainingClient.TrainProjectAsync(projectId);
            Console.WriteLine("Training");

            // Poll to check the training iteration status
            while (true)
            {
                iteration = await m_TrainingClient.GetIterationAsync(projectId, iteration.Id);
                Console.Write(".");

                if (iteration.Status != "Training")
                {
                    break;
                }
                Thread.Sleep(1000);
            }
            Console.WriteLine("Complete!");

            // Display the training time
            var trainingTime = iteration.TrainedAt - iteration.Created;
            Console.WriteLine($"Training time: { trainingTime.Value.ToString(@"hh\:mm\:ss") }");
            Console.WriteLine();
            return iteration;
        }

        public async Task DisplayStatistics(Guid projectId, Iteration iteration)
        {
            Console.WriteLine("Image performance statistics");
            var temp = Console.ForegroundColor;

            Console.WriteLine("Iteration performance statistics");
            var iterationPerformanceStatistics = 
                await m_TrainingClient.GetIterationPerformanceAsync(projectId, iteration.Id);

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"    AveragePrecision:      { iterationPerformanceStatistics.AveragePrecision }");
            Console.WriteLine($"    Precision:             { iterationPerformanceStatistics.Precision }");
            Console.WriteLine($"    PrecisionStdDeviation: { iterationPerformanceStatistics.PrecisionStdDeviation }");
            Console.WriteLine($"    Recall:                { iterationPerformanceStatistics.Recall }");
            Console.WriteLine($"    RecallStdDeviation:    { iterationPerformanceStatistics.RecallStdDeviation }");
            Console.WriteLine();

            Console.ForegroundColor = temp;
            Console.WriteLine("Image performance statistics");
            var imagePerformanceStatistics = 
                await m_TrainingClient.GetImagePerformancesAsync(projectId, iteration.Id);

            foreach (var performanceStatistic in imagePerformanceStatistics)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"    { performanceStatistic.Tags[0].TagName }");

                foreach (var prediction in performanceStatistic.Predictions)
                {
                    Console.ForegroundColor = 
                        performanceStatistic.Tags[0].TagName == prediction.TagName ? ConsoleColor.Green : ConsoleColor.Red;
                    Console.WriteLine($"        Tag: { prediction.TagName } - Probability: { (int)(prediction.Probability * 100) }%");
                }
            }
            Console.WriteLine();


            Console.ForegroundColor = temp;
            Console.WriteLine("     Per Tag Performance");
            foreach (var tagPerformacne in iterationPerformanceStatistics.PerTagPerformance)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"        { tagPerformacne.Name }");
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"            Precision:             { tagPerformacne.Precision }");
                Console.WriteLine($"            PrecisionStdDeviation: { tagPerformacne.PrecisionStdDeviation }");
                Console.WriteLine($"            Recall:                { tagPerformacne.Recall }");
                Console.WriteLine($"            RecallStdDeviation:    { tagPerformacne.RecallStdDeviation }");
            }
            Console.WriteLine();

            Console.ForegroundColor = temp;

        }

        public void Dispose()
        {
            m_TrainingClient.Dispose();
        }



        private async Task TagTestImages(Guid projectId, string folder, string tagName)
        {

            // Create the tag and get the ID
            var tag = await m_TrainingClient.CreateTagAsync(projectId, tagName);
            var tagIds = new List<Guid> { tag.Id };

            // Get a list of the training images in the folder.
            var files = Directory.GetFiles(folder);

            int count = 0;
            var images = new List<ImageFileCreateEntry>();
            foreach (var file in files)
            {
                // Cerate an ImageFileCreateEntry for the file
                var imageFileCreateEntry = new ImageFileCreateEntry
                {
                    Name = Path.GetFileName(file),
                    Contents = File.ReadAllBytes(file)
                };
                images.Add(imageFileCreateEntry);
                count++;
                if (count == ImagesPerClass)
                {
                    break;
                }

                // The maximum ImageFileCreateEntry size is 64
                if (images.Count == 64)
                {
                    // Create a batch, execute it and clear the list
                    var batch = new ImageFileCreateBatch
                    {
                        Images = images,
                        TagIds = tagIds
                    };
                    await m_TrainingClient.CreateImagesFromFilesAsync(projectId, batch);
                    images.Clear();
                }

            }

            // Create a batch and execute it
            if (images.Count > 0)
            {
                var batch = new ImageFileCreateBatch
                {
                    Images = images,
                    TagIds = tagIds
                };
                await m_TrainingClient.CreateImagesFromFilesAsync(projectId, batch);
            }

        }



        private List<string> GetFolderNames(string path, int minImageCount, int numberOfClasses)
        {
            var folderNames = new List<string>();

            // Iterate through the test folders
            var trainingImageFolders = Directory.EnumerateDirectories(path);
            foreach (var trainingFolder in trainingImageFolders)
            {
                // Check the image count
                if (Directory.GetFiles(trainingFolder).Length >= minImageCount)
                {
                    folderNames.Add(Path.GetFileName(trainingFolder));
                    if (folderNames.Count == numberOfClasses)
                    {
                        break;
                    }
                }
            }

            return folderNames;
        }


    }
}
