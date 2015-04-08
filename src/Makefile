CXXFLAGS = -O3 -g -Wall
LDLIBS = -lopencv_core -lopencv_highgui

motiontracking: motiontracking.cpp helper_functions.cpp helper_functions.h
	c++ -O3 -g -Wall    motiontracking.cpp helper_functions.cpp -lopencv_core -lopencv_highgui -o motiontracking

.PHONY:
clean:
	rm -rf motiontracking *.o

