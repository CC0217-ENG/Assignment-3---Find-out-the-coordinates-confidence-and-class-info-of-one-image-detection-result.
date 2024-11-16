from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput
#import jetson.inference
#import jetson.utils

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = videoSource("/dev/video0")	#'/dev/video0' for V4L2
display = videoOutput("display://0")	#'my_video.mp4' for file

while display.IsStreaming():		#main loop will go here
	img = camera.Capture()
	if img is None: # capture timeout
		continue
	detections = net.Detect(img)

	print(f"box num: {len(detections)}")
	for idx, d in enumerate(detections):
		print(f"\n\nbox index: {idx}\n\n")
		print(detections[idx])

	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
	
	
