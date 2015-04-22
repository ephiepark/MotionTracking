CXXFLAGS = -O3 -g -Wall
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video

motiontracking: motiontracking.cu helper_functions.h
	nvcc -o motiontracking motiontracking.cu $(LDLIBS) -arch=sm_30

.PHONY:
clean:
	rm -rf motiontracking *.o

