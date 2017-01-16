using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Convolution
{
    internal static class BitmapUtility
    {
        private const PixelFormat DefaultPixelFormat = PixelFormat.Format24bppRgb;

        internal static Image FromColorArray(ColorRaw[] data, int width, int height)
        {
            var pinned = GCHandle.Alloc(data, GCHandleType.Pinned);
            var result = new Bitmap(width, height, width * Marshal.SizeOf<ColorRaw>(), DefaultPixelFormat, pinned.AddrOfPinnedObject());
            pinned.Free();

            return result;
        }

        internal static ColorRaw[] ToColorArray(Bitmap bmp)
        {
            var result = new ColorRaw[bmp.Width * bmp.Height];
            var temp = new byte[Marshal.SizeOf<ColorRaw>() * bmp.Width * bmp.Height];
            var data = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, DefaultPixelFormat);

            Marshal.Copy(data.Scan0, temp, 0, temp.Length);
            bmp.UnlockBits(data);

            Parallel.For(0, result.Length, i =>
            {
                var k = 3 * i;

                result[i] = new ColorRaw
                {
                    R = temp[k + 0],
                    G = temp[k + 1],
                    B = temp[k + 2],
                };
            });

            return result;
        }
    }
}