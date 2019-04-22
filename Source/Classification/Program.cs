using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.FileProperties;
using Windows.Storage.Search;
using Windows.Storage.Streams;
using System.Linq;
using Windows.AI.MachineLearning;

namespace DisneyCastleClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Running...");

            Run().ContinueWith((_) => Console.WriteLine("Done!"));

            Console.ReadKey();
        }

        private static async Task Run()
        {
            StorageFile modelFile = 
                await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/ChocolatOrRaisin.onnx"));
            ChocolatOrRaisinModel model = await ChocolatOrRaisinModel.CreateFromStreamAsync(modelFile as IRandomAccessStreamReference);

            foreach (StorageFile image in await GetImagesFromExecutingFolder())
            {
                Console.Write($"Processing {image.Name}... ");
                var videoFrame = await GetVideoFrame(image);

                ChocolatOrRaisinInput ModelInput = new ChocolatOrRaisinInput();
                ModelInput.data = ImageFeatureValue.CreateFromVideoFrame(videoFrame);

                var ModelOutput = await model.EvaluateAsync(ModelInput);
                var topCategory = ModelOutput.loss.SelectMany(l => l).OrderByDescending(kvp => kvp.Value).FirstOrDefault();
                var label = ModelOutput.classLabel.GetAsVectorView().ToList().FirstOrDefault();

                Console.Write($"DONE ({topCategory.Key} {topCategory.Value:P2}) - ClassLabel => {label} \n");

                await UpdateImageTagMetadata(await image.Properties.GetImagePropertiesAsync(), topCategory.Key);
            }
        }

        private static async Task<IEnumerable<StorageFile>> GetImagesFromExecutingFolder()
        {
            var folder = await StorageFolder.GetFolderFromPathAsync(Environment.CurrentDirectory);

            var queryOptions = new QueryOptions(CommonFileQuery.DefaultQuery, new List<string>() { ".jpg", ".png" });

            return await folder.CreateFileQueryWithOptions(queryOptions)?.GetFilesAsync();
        }

        private static async Task UpdateImageTagMetadata(ImageProperties imageProperties, params string[] tags)
        {
            var propertiesToSave = new List<KeyValuePair<string, object>>()
            {
                new KeyValuePair<string, object>("System.Keywords", String.Join(';', tags))
            };
            await imageProperties.SavePropertiesAsync(propertiesToSave);
        }

        private static async Task<VideoFrame> GetVideoFrame(StorageFile file)
        {
            SoftwareBitmap softwareBitmap;
            using (IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read))
            {
                // Create the decoder from the stream 
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                // Get the SoftwareBitmap representation of the file in BGRA8 format
                softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);

                return VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);
            }
        }
    }
}