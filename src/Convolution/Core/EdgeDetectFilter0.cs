using System;
using System.Drawing;
using System.Threading.Tasks;
using Alea;
using Alea.CSharp;

namespace Convolution
{
    // Todo: Check this out: https://www.evl.uic.edu/sjames/cs525/final.html
    internal static class EdgeDetectFilter0
    {
        // CPU: Using Native GDI+ Bitmap!
        internal static Image RenderCpu1(Bitmap image)
        {
            var width = image.Width;
            var height = image.Height;

            var source = new FastBitmap(image);
            var result = new FastBitmap(image.Width, image.Height);
            var matrix = PixelMatrix.EdgeDetectFilter0;

            Parallel.For(0, source.Height, y =>
            {
                for (var x = 0; x < source.Width; ++x)
                {
                    result.SetPixel(x, y, matrix.Compute(GetNeighbours(source, x, y, width, height)));
                }
            });

            return result.Bitmap;
        }

        // CPU: Using byte Array!
        internal static Image RenderCpu2(Bitmap image)
        {
            var width = image.Width;
            var height = image.Height;

            var source = FastBitmap.ToColorArray(image);
            var result = new ColorRaw[width * height];
            var matrix = PixelMatrix.EdgeDetectFilter0;

            Parallel.For(0, height, y =>
            {
                for (var x = 0; x < width; ++x)
                {
                    var offset = y * width + x;
                    result[offset] = PixelMatrix.ComputeRaw(GetNeighboursRaw(source, x, y, width, height), matrix.Filter,
                        matrix.Factor, matrix.Offset);
                }
            });

            return FastBitmap.FromColorArray(result, image.Width, image.Height);
        }

        // GPU: Using byte Array!
        [GpuManaged]
        internal static Image RenderGpu1(Bitmap image)
        {
            var width = image.Width;
            var height = image.Height;

            var source = FastBitmap.ToColorArray(image);
            var result = new ColorRaw[width * height];
            var matrix = PixelMatrix.EdgeDetectFilter0;
            var mFilter = matrix.Filter;
            var mfactor = matrix.Factor;
            var mOffset = matrix.Offset;
            var lp = ComputeLaunchParameters(width, height);

            Gpu.Default.Launch(() =>
            {
                var x = blockDim.x * blockIdx.x + threadIdx.x;
                var y = blockDim.y * blockIdx.y + threadIdx.y;

                var offset = y * width + x;

                if (offset < result.Length)
                {
                    int sy = y - 1;
                    int sx = x - 1;
                    int yl = y + 1;
                    int xl = x + 1;

                    float r = 0;
                    float g = 0;
                    float b = 0;

                    int counter = 0;

                    for (int ny = sy; ny <= yl; ++ny)
                    {
                        for (int nx = sx; nx <= xl; ++nx)
                        {
                            var oOffset = ny * width + nx;

                            var pixelData = oOffset < result.Length
                                ? source[oOffset]
                                : ColorRaw.FromRgb(0, 0, 0);

                            var rx = pixelData.R;
                            var gx = pixelData.G;
                            var bx = pixelData.B;

                            r += rx * mFilter[counter];
                            g += gx * mFilter[counter];
                            b += bx * mFilter[counter];

                            //r += mFilter[counter] * pixelData.R;
                            //g += mFilter[counter] * pixelData.G;
                            //b += mFilter[counter] * pixelData.B;

                            ++counter;
                        }
                    }

                    r = r / mfactor + mOffset;
                    g = g / mfactor + mOffset;
                    b = b / mfactor + mOffset;

                    result[offset] = ColorRaw.FromRgb((byte)r, (byte)g, (byte)b);
                }
            }, lp);

            return FastBitmap.FromColorArray(result, image.Width, image.Height);
        }

        //// GPU: Parallel.For!
        //internal static Image RenderGpu3(Bitmap image)
        //{
        //    var result = FastBitmap.ToByteArray(image);

        //    Gpu.Default.For(0, image.Width * image.Height, i =>
        //    {
        //        var offset = 3 * i;
        //        ComputeEdgeDetectFilter0AtOffset(result, offset);
        //    });

        //    return FastBitmap.FromByteArray(result, image.Width, image.Height);
        //}

        private static LaunchParam ComputeLaunchParameters(int width, int height)
        {
            const int threads = 32;
            return new LaunchParam(new dim3((width + (threads - 1)) / threads, (height + (threads - 1)) / threads),
                new dim3(threads, threads));
        }

        private static Color[] GetNeighbours(FastBitmap source, int x, int y, int width, int height)
        {
            const int offset = 1;

            int sy = y - offset;
            int sx = x - offset;
            int yl = y + offset;
            int xl = x + offset;

            Color[] pixelData = new Color[9];

            int counter = 0;

            for (int i = sy; i <= yl; ++i)
            {
                for (int k = sx; k <= xl; ++k)
                {
                    pixelData[counter++] = IsValid(k, i, width, height) == false
                        ? Color.FromArgb(0, 0, 0, 0)
                        : source.GetPixel(k, i);
                }
            }

            return pixelData;
        }

        // Todo: This needs cache and it's begging for a Texture in terms of memory access.
        // ReSharper disable once SuggestBaseTypeForParameter
        private static ColorRaw[] GetNeighboursRaw(ColorRaw[] source, int x, int y, int width, int height)
        {
            const int offset = 1;

            int sy = y - offset;
            int sx = x - offset;
            int yl = y + offset;
            int xl = x + offset;

            var pixelData = new ColorRaw[9];

            int counter = 0;

            for (int ny = sy; ny <= yl; ++ny)
            {
                for (int nx = sx; nx <= xl; ++nx)
                {
                    pixelData[counter++] = IsValid(nx, ny, width, height) == false
                        ? ColorRaw.FromRgb(0, 0, 0)
                        : source[ny * width + nx];
                }
            }

            return pixelData;
        }

        private static bool IsValid(int x, int y, int width, int height)
        {
            return x >= 0 && x < width && y >= 0 && y < height;
        }
    }
}