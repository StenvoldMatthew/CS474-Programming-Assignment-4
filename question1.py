import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
#from scipy.fftpack import fft2, ifft2, fftshift

def showImages(images, titles):
    # Determine the number of images
    num_images = len(images)
    
    # Calculate number of rows and columns
    if num_images > 4:
        num_rows = (num_images // 4) + (num_images % 4 > 0)  # Add an extra row if there are leftovers
        num_cols = 4
    else:
        num_rows = 1
        num_cols = num_images

    # Create subplots with the calculated rows and columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each image
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')

    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_img = 255 * (image - min_val) / (max_val - min_val)
    return norm_img.astype(np.uint8)

def pad_image(image, pad_height, pad_width):
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

def getGaussianFilter(size):
    # Getting this were harder than the coding assignment
    array = None
    if size == 7:
        array =  [[1, 1, 2, 2, 2, 1, 1],
                  [1, 2, 2, 4, 2, 2, 1],
                  [2, 2, 4, 8, 4, 2, 2],
                  [2, 4, 8, 16, 8, 4, 2],
                  [2, 2, 4, 8, 4, 2, 2],
                  [1, 2, 2, 4, 2, 2, 1],
                  [1, 1, 2, 2, 2, 1, 1]]
    if size == 15:
        array =  [[2, 2, 3, 4, 5, 5, 6, 6, 6, 5, 5, 4, 3, 2, 2],
                  [2, 3, 4, 5, 7, 7, 8, 8, 8, 7, 7, 5, 4, 3, 2],
                  [3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4, 3],
                  [4, 5, 7, 9, 10, 12, 13, 13, 13, 12, 10, 9, 7, 5, 4],
                  [5, 7, 9, 11, 13, 14, 15, 16, 15, 14, 13, 11, 9, 7, 5],
                  [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
                  [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
                  [6, 8, 11, 13, 16, 18, 19, 20, 19, 18, 16, 13, 11, 8, 6],
                  [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
                  [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
                  [5, 7, 9, 11, 13, 14, 15, 16, 15, 14, 13, 11, 9, 7, 5],
                  [4, 5, 7, 9, 10, 12, 13, 13, 13, 12, 10, 9, 7, 5, 4],
                  [3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4, 3],
                  [2, 3, 4, 5, 7, 7, 8, 8, 8, 7, 7, 5, 4, 3, 2],
                  [2, 2, 3, 4, 5, 5, 6, 6, 6, 5, 5, 4, 3, 2, 2],]
        
    npArray = np.array(array, dtype=np.float32)
    # Make it so the array adds up to 1
    return npArray / np.sum(npArray)

def doGaussian(filename, maskSize):
    mask = getGaussianFilter(maskSize)
    image = np.array(Image.open(filename).convert('L'), dtype=np.float32)
    paddedImage = pad_image(image, maskSize // 2, maskSize // 2)
    output_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = paddedImage[i:i + maskSize, j:j + maskSize]
            output_image[i, j] = np.sum(region * mask)
    
    return output_image

def fft(data: list[float], nn: int, isign: int):
  #n, mmax, m, j, istep, i = None # uint32
  #wtemp, wr, wpr, wpi, wi, theta = None # double
  #tempr, tempi = None # float

  n = nn*2
  j = 1
  for i in (range(1, n, 2)):
    if j > i:
      data[j], data[i] = data[i], data[j]
      data[j+1], data[i+1] = data[i+1], data[j+1]
    m = nn
    while (m >= 2 and j > m):
      j -= m
      m //= 2
    j += m

  mmax = 2
  while (n > mmax):
    istep = mmax *2
    theta = isign * (2*np.pi / mmax)
    wtemp = np.sin(.5*theta)
    wpr = -2.0 * wtemp * wtemp
    wpi = np.sin(theta)
    wr = 1.0
    wi = 0.0
    for m in range(1, mmax, 2):
      for i in range(m, n+1, istep):
        j = i + mmax
        tempr = wr*data[j]-wi*data[j+1]
        tempi = wr*data[j+1]+wi*data[j]
        data[j] = data[i]-tempr
        data[j+1] = data[i+1]-tempi
        data[i] += tempr
        data[i+1] += tempi

      wtemp = wr
      wr = wtemp*wpr-wi*wpi+wr
      wi = wi*wpr+wtemp*wpi+wi
    mmax=istep

def convertToInput2d(arr, N, M):
  data = np.zeros((N*2+1, M*2+1))
  for i in range(N):
    for j in range(M):
      data[2 * i + 1, 2* j + 1] = arr[i, j]  # Real
  return data

def fft2d(data, isign):
  if isign == -1:
    data = convertToInput2d(data, data.shape[0], data.shape[1])
  else:
    data /= (data.shape[0]//2)*(data.shape[1]//2)
  
  # Each pixel is represented by the following matrix: | R  I |
  # The I values are set to the most recently          | I  0 |
  # updated value before computing

  # Apply 1D FFT on rows
  for i in range(1, data.shape[0], 2):
    data[i, 2::2] = data[i+1, 1::2] 
    fft(data[i, :], data.shape[1]//2, isign)

  # Apply 1D FFT on columns
  for j in range(1, data.shape[1], 2):
    data[2::2, j] = data[1::2, j+1]
    fft(data[:, j], data.shape[0]//2, isign)

  if isign == 1:
    return data[1::2, 1::2]
  return data

def extractResults(data):
  real = data[1::2, 1::2]
  imag = data[2::2, 1::2]
  mag = np.zeros_like(real)
  for i in range(real.shape[0]):
    for j in range(real.shape[1]):   
      mag[i,j] = np.sqrt(real[i,j]**2 + imag[i,j]**2)
  return real, imag, mag

def combineResultes(real, imag):
   data = np.zeros((2 * real.shape[0] + 1, 2 * real.shape[1] + 1))
   data[1::2, 1::2] = real
   data[2::2, 1::2] = imag

   return data

def fftShift(data):
    real, imag, mag = extractResults(data)

    def shift_quadrants(data):
        rows, cols = data.shape
        mid_row, mid_col = rows // 2, cols // 2

        shifted = np.zeros_like(data)
        shifted[:mid_row, :mid_col] = data[mid_row:, mid_col:]  # Bottom-right -> Top-left
        shifted[:mid_row, mid_col:] = data[mid_row:, :mid_col]  # Bottom-left -> Top-right
        shifted[mid_row:, :mid_col] = data[:mid_row, mid_col:]  # Top-right -> Bottom-left
        shifted[mid_row:, mid_col:] = data[:mid_row, :mid_col]  # Top-left -> Bottom-right

        return shifted

    # Apply the same shift logic to both real and imaginary parts
    shifted_real = shift_quadrants(real)
    shifted_imag = shift_quadrants(imag)

    return shifted_real, shifted_imag


def band_reject_filter(filename, low_freq, high_freq):
    image = np.array(Image.open(filename).convert('L'), dtype=np.float32)
    # Compute FFT of the image
    f_image = fft2d(image, -1)
    shiftedReal, shiftedImag = fftShift(f_image)

    # Create a mask for the band-reject filter
    mask = np.ones(image.shape)
    rows, cols = image.shape
    center = (rows // 2, cols // 2)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if low_freq <= dist <= high_freq:
                mask[i, j] = 0

    # Apply the mask to the Fourier transform
    shiftedReal *= mask
    shiftedImag *= mask
    fixedShiftedFFT = combineResultes(shiftedReal, shiftedImag)
    temp1, temp2 = fftShift(fixedShiftedFFT)
    fixedImageFFT = combineResultes(temp1, temp2)

    # Inverse FFT
    filtered_image = np.real(fft2d(fixedImageFFT, 1))

    return filtered_image

def notch_filter(filename, radius):
    image = np.array(Image.open(filename).convert('L'), dtype=np.float32)
    # Compute FFT of the image
    f_image = fft2d(image, -1)
    shiftedReal, shiftedImag = fftShift(f_image)

    # Create a mask for the notch filter
    mask = np.ones(image.shape)
    rows, cols = image.shape

    center = (rows // 2, cols // 2)
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask[dist <= radius] = 0

    # Apply the mask to the Fourier transform
    shiftedReal *= mask
    shiftedImag *= mask
    fixedShiftedFFT = combineResultes(shiftedReal, shiftedImag)
    temp1, temp2 = fftShift(fixedShiftedFFT)
    fixedImageFFT = combineResultes(temp1, temp2)

    # Inverse FFT
    filtered_image = np.real(fft2d(fixedImageFFT, 1))

    return filtered_image


def runProgram(filename):
  images = []
  titles = []
  image = np.array(Image.open(filename).convert('L'), dtype=np.float32)

  images.append(image)
  titles.append("Original")

  lowFreq, highFreq = 34, 36
  bandImage = band_reject_filter(filename, lowFreq, highFreq)
  images.append(bandImage)
  titles.append("Band-Reject")
  
  notch_radius = 35
  noiseOfImage = notch_filter(filename, notch_radius)
  images.append(image - noiseOfImage)
  titles.append("Notch")

  images.append(noiseOfImage)
  titles.append("Isolating Noise")
  
  gaussianImage = doGaussian(filename, 15)
  images.append(gaussianImage)
  titles.append("Gaussian Comparison")
  
  showImages(images, titles)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Converter')
    parser.add_argument('-f','--filename', type=str, default = "boy_noisy.png", help='Which file do you want to use')
    args = parser.parse_args()

    runProgram(args.filename)