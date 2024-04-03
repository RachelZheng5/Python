import spectral
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pathlib
import HSI_Code.directory_data_process as fp

fig_o_dir = '/Users/rachel/HSI_Project/HSI_files/output'  
img_i_dir = '/Users/rachel/HSI_Project/HSI_files/input/'


def compareWavelengths(img1, img2, img3, img_path1, img_path2, img_path3):
    # Select which pixels to display
    pixels_of_interest = [(15,0), (35,0)]
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    # Display spectral data of pixels
    for x, y in pixels_of_interest:
        wavelength1 = img1.read_pixel(x, y)
        wavelength2 = img2.read_pixel(x, y)
        wavelength3 = img3.read_pixel(x, y)

        plt.plot(np.arange(len(wavelength1)), wavelength1, label=f'Pixel at ({x}, {y}) on {img_path1[78:130]}')
        plt.plot(np.arange(len(wavelength2)), wavelength2, label=f'Pixel at ({x}, {y}) on {img_path2[78:130]}')
        plt.plot(np.arange(len(wavelength3)), wavelength3, label=f'Pixel at ({x}, {y}) on {img_path3[78:130]}')

  

    plt.title(img_path1[114:117] + 'v' + img_path2[114:117] + 'v' + img_path3[114:117])
    plt.xlabel('Wavelength')
    plt.ylabel('Data Value')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig_o_name = 'WavelengthComp_' + img_path1[116:117] + 'v' + img_path2[116:117] + 'v' + img_path3[116:117] + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    fig_o_path = pathlib.PurePath(fig_o_dir, fig_o_name)
    fig.savefig(fig_o_path)

def write_all_pixel_values_to_txt(img, img_path, filename):
    imgName = os.path.basename(img_path)
    with open(filename, 'a') as f:
        f.write(imgName +'\n')
        height, width, channels = img.shape  # Assuming img is a numpy array
        for y in range(height):
            for x in range(width):
                pixel_values = img[y, x]  # Get pixel values at (y, x) position
                pixel_str = ' '.join(str(val) for val in pixel_values)
                f.write(pixel_str + '\n')
        f.write('\n\n\n')

def plottingWavelengths(img):
    width, height, num_bands = img.shape
    pixel_values = img.load()

    num_indices = 25
    step_size = max(1, width * height // num_indices) #using floor division to calculate step_size

    for band in range(1):
        plt.figure(figsize=(10, 6))
        
        # Flatten 2d pixel values and chosing every step_size value
        band_values = pixel_values[band, :, :].flatten()[::step_size]
        pixel_indices = np.linspace(0, width * height - 1, len(band_values), dtype=int)

        plt.plot(pixel_indices, band_values, marker='.', linestyle='-')
        plt.title(f'Band {band + 1} - Pixel Intensity')
        plt.xlabel('Pixel Index')
        plt.ylabel('Pixel Value')
        plt.grid(True)
        plt.show()


dataset_filename = 'processingDatasetFile.txt'
def write_dataset_to_txt(dataset_filename, dataset):
    np.savetxt(dataset_filename, dataset, delimiter=',', fmt='%s')


def create_ordered_pairs(img_width, img_height, increment=5):
    ordered_pairs = []
    for i in range(1):
        x, y = 0,0
        while y < img_height:
            ordered_pairs.append((x, y))
            x += increment
            y += increment

    return ordered_pairs

def create_dataset(folder_path, num_samples=10, num_pixels_per_sample=10):
    count = row = col = pImgCounter= 0
    num_rows = 380
    num_cols = 467
    dataset = np.empty((num_rows, num_cols), dtype=object)

    #header
    for col in range(462):
        dataset[0, col] = f'BandNo{col}'

    dataset[0, 462] = f'sidx'
    dataset[0, 463] = f'pidx'
    dataset[0, 464] = f'category'
    dataset[0, 465] = f'pName'
    dataset[0, 466] = f'sName'

    pairs = create_ordered_pairs(50, 50, increment=5)

    count = 0 
    for root, dirs, files in os.walk(img_i_dir):
        for file_name in files:
            if (file_name.endswith('.hdr') & count < 5):
                full_path_hdr = os.path.join(root, file_name)
                # full_path_data = os.path.splitext(full_path_hdr)[0] + '.hdr'  # Assuming data file has .dat extension
                print(full_path_hdr)
                #img = spectral.open_image(full_path_data)
                #height, width, bands = img.shape
                #print(f"Height: {height}, Width: {width}, Bands: {bands}")
            print()
            count +=1

        #         for pair in pairs: 
        #             wavelength = img.read_pixel(pair[0], pair[1])

        #             dataset[row, 462] = file_name[20:]
        #             dataset[row, 463] = pImgCounter
        #             dataset[row, 464] = file_name[20:]
        #             dataset[row, 465] = root[70:]
        #             dataset[row, 466] = file_name[20:]

        #             for col, wave_value in enumerate(wavelength):
        #                 dataset[row, col] = wave_value
        # pImgCounter+=1
    write_dataset_to_txt(dataset_filename, dataset)
    return dataset

    

def main():
    img_i_dir = '/Users/rachel/HSI_Project/HSI_files/input/'

    # Read hyperspectral image
    img_path1 = '/Users/rachel/HSI_Project/HSI_files/input/Scan_06-10-2022_0941_AZ270_EL30_G_D/Scan_06-10-2022_0941_AZ270_EL30_G_D_c01_x0514y0165w50h50.hdr'
    img_path2 = '/Users/rachel/HSI_Project/HSI_files/input/Scan_07-11-2022_1020_AZ000_EL30_L_D/Scan_07-11-2022_1020_AZ000_EL30_L_D_c05_x1467y2397w50h50.hdr'
    img_path3 = '/Users/rachel/HSI_Project/HSI_files/input/Scan_07-13-2022_0942_AZ225_EL15_L_D/Scan_07-13-2022_0942_AA225_EL15_L_D_c05_x1087y3119w50h50.hdr'
    
    img1 = spectral.open_image(img_path1)
    img2 = spectral.open_image(img_path2)
    img3 = spectral.open_image(img_path3)

    #compareWavelengths(img1,img2,img3, img_path1, img_path2, img_path3)
    #obtaining_pixel_values_for_training(img1,img_path1, img2,img_path2, img3, img_path3)
    #plottingWavelengths(img1)
    dataset = create_dataset(img_i_dir, 10,10)
    # print(dataset.shape)
    # print(dataset)


if __name__ == '__main__':
    main()
#--------GOAL
# dimentionality reduction to collapse data into smaller dataset
# pca to get eigenvectors ( i.e. data that are unique)
# train a dataset that takes into account wavelengths, average varience of wavelengths, 