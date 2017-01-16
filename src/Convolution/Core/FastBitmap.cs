using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace Convolution
{
    // Ignore the fact that we've not implemented IDisposable
    internal sealed class FastBitmap
    {
        private const PixelFormat PixelFormat = System.Drawing.Imaging.PixelFormat.Format24bppRgb;

        private readonly Bitmap _bitmap;
        private readonly BitmapData _data;

        internal FastBitmap(int width, int height)
        {
            Width = width;
            Height = height;
            _bitmap = new Bitmap(width, height, PixelFormat);
            _data = _bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadWrite, PixelFormat);
        }

        internal FastBitmap(Bitmap image)
        {
            Width = image.Width;
            Height = image.Height;
            _bitmap = image.Clone(new Rectangle(0, 0, image.Width, image.Height), PixelFormat);
            _data = _bitmap.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadWrite, PixelFormat);
        }

        // Todo: Fix!
        //~FastBitmap()
        //{
        //    _bitmap.UnlockBits(_data);
        //}

        internal static Image FromByteArray(byte[] data, int width, int height)
        {
            var pinnedData = GCHandle.Alloc(data, GCHandleType.Pinned);
            var result = new Bitmap(width, height, 3 * width * sizeof(byte), PixelFormat, pinnedData.AddrOfPinnedObject());

            pinnedData.Free();
            return result;
        }

        internal static byte[] ToByteArray(Bitmap bmp)
        {
            var data = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat);

            var result = new byte[3 * bmp.Width * bmp.Height];
            Marshal.Copy(data.Scan0, result, 0, result.Length);

            bmp.UnlockBits(data);

            return result;
        }

        internal int Width { get; }

        internal int Height { get; }

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