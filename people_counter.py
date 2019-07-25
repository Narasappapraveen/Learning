from peopleCounterModule import PeopleCounterModule
import argparse

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.3,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=20,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	pc = PeopleCounterModule(args)
	pc.run()

main()
