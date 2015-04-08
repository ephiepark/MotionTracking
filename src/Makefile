CXXFLAGS = -O3 -g -Wall
LDLIBS = -lopencv_core -lopencv_highgui

motiontracking: helper_functions.o

helper_functions.o: helper_functions.h

.PHONY:
clean:
	rm -rf motiontracking *.o

