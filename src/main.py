import os
import numpy as np
import cv2
import natsort
from Color_Balance_and_fusion.color_balance_and_fusion import color_balance_and_fusion
from Wavelength_Compensation_and_Dehazing.wavelength_compensation_and_dehazing import wavelength_compensation_and_dehazing
from Low_Complexity_Enhancement_Based_on_Dark_Channel_Prior.low_complexity import low_complexity
from Integrated_Color_Model.integrated_color_model import integrated_color_model

def main():
    input_path = "./InputImages"
    output_path = "./OutputImages/"
    files = os.listdir(input_path)
    files = natsort.natsorted(files)

    for i in range(len(files)):
        file = files[i]
        file_path = input_path + "/" + file
        prefix = file.split('.')[0]
        if os.path.isfile(file_path):
            print("****** file ******", file)
            img = cv2.imread(file_path)
            output1 = color_balance_and_fusion(img)
            output2 = wavelength_compensation_and_dehazing(img)
            output3 = low_complexity(img)
            output4 = integrated_color_model(img)
            cv2.imwrite(output_path + "Color_Balance_and_fusion/" + prefix + ".png", output1)
            cv2.imwrite(output_path + "Wavelength_Compensation_and_Dehazing/" + prefix + ".png", output2)
            cv2.imwrite(output_path + "Low_Complexity_Enhancement_Based_on_Dark_Channel/" + prefix + ".png", output3)
            cv2.imwrite(output_path + "Integrated_Color_Model/" + prefix + ".png", output4)
if __name__ == "__main__":
    main()