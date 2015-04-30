CXXFLAGS = -O3 -g -Wall
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video

motiontracking: motiontracking.cu motiontracking_cpu.cu helper_functions.h
	nvcc -O3 -o motiontracking motiontracking.cu $(LDLIBS) -arch=sm_30
	nvcc -O3 -o cpu_motiontracking motiontracking_cpu.cu $(LDLIBS) -arch=sm_30

.PHONY:
clean:
	rm -rf motiontracking *.o

