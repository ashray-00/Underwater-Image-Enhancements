all:
	python3 src/main.py
	python3 src/UCIQE.py
	python3 src/UIQE.py

enhance:
	python3 src/main.py

metric:
	python3 src/UCIQE.py
	python3 src/UIQE.py

clean:
	rm OutputImages/Blue_green_channels_dehazing_and_red_channel_correction/*.png
	rm OutputImages/Color_Balance_and_fusion/*.png
	rm OutputImages/Integrated_Color_Model/*.png
	rm OutputImages/Low_Complexity_Enhancement_Based_on_Dark_Channel/*.png
	rm Metric_Output/*.csv
