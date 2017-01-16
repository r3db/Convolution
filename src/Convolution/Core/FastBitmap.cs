using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace Convolution
{
    // Ignore the fact that we've not implemented IDisposable
    internal sealed class FastBitmap
    {
        private const PixelFormat DefaultPixelFormat = PixelFormat.Format24bppRgb;

        private readonly Bitmap _bitmap;
        private readonly BitmapData _data;

        internal FastBitmap(int width, int height)
        {
            _bitmap = new Bitmap(width, height, DefaultPixelFormat);
            _data = _bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, DefaultPixelFormat);
        }

        internal FastBitmap(Bitmap image)
        {
            _bitmap = image.Clone(new Rectangle(0, 0, image.Width, image.Height), DefaultPixelFormat);
            _data = _bitmap.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadWrite, DefaultPixelFormat);
        }

        internal unsafe Color GetPixel(int x, int y)
        {
            var pixel = (byte*)_data.Scan0.ToPointer() + (y * _data.Stride + 3 * x);
            return Color.FromArgb(pixel[2], pixel[1], pixel[0]);
        }

        internal unsafe void SetPixel(int x, int y, Color color)
        {
            var pixel = (byte*)_data.Scan0.ToPointer() + (y * _data.Stride + 3 * x);

            pixel[0] = color.B;
            pixel[1] = color.G;
            pixel[2] = color.R;
        }

        internal Bitmap Bitmap => _bitmap;
    }
}