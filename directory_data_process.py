import data_txt_creation as dtc
import numpy as np
import os
import spectral

img_i_dir = '/Users/rachel/HSI_Project/HSI_files/input/'

def obtainingFromDirectory():
    num_rows = 2001
    num_cols = 467
    dataset = np.empty((num_rows, num_cols), dtype=object)
    row = col = pImgCounter = root_iterator = file_iterator = 0
    row = sImgCounter = pImgCounter = 1

    #header
    for col in range(462):
        dataset[0, col] = f'BandNo{col}'

    dataset[0, 462] = f'sidx'
    dataset[0, 463] = f'pidx'
    dataset[0, 464] = f'category'
    dataset[0, 465] = f'pName'
    dataset[0, 466] = f'sName'


    pairs = dtc.create_ordered_pairs(img_width = 50, img_height=50, increment=5)
    directory = np.empty(380, dtype=object)
    rootData = np.empty(38, dtype=object)
    

    for root, dirs, files in os.walk(img_i_dir):
        # rootData[root_iterator] = root
        # root_iterator+=1
        # print(root)
        for file_name in files:
            if(file_name.endswith('.hdr')):
                directory[file_iterator] = os.path.join(root, file_name)
                img = spectral.open_image(directory[file_iterator])
                file_iterator+=1

                for pair in pairs: 
                    wavelength = img.read_pixel(pair[0], pair[1])
                    print(file_name[78:])
                    dataset[row, 462] = 'S0' + str(sImgCounter)
                    dataset[row, 463] = 'P0' + str(pImgCounter)
                    dataset[row, 464] = file_name[36:39]
                    dataset[row, 465] = root[42:]
                    dataset[row, 466] = file_name[:]

                    for col, wave_value in enumerate(wavelength):
                        dataset[row, col] = wave_value
    
                    row+= 1
                    if(row == num_rows):
                        break
                    sImgCounter+=1

                if(row == num_rows):
                    break
                pImgCounter+=1

            if(row == num_rows):
                break
        if(row == num_rows):
                break

    dtc.write_dataset_to_txt('directory_dataset.txt', dataset)
    print(dataset[:5, :])  # first 5 rows 

    # dtc.write_dataset_to_txt("directory_image_path.txt", directory)
    # dtc.write_dataset_to_txt("root_path.txt", rootData)

 

def main():
    obtainingFromDirectory()

if __name__ == '__main__':
    main()
