CPP_FLAGS=-std=c++11 -Wall -g
LD_FLAGS=`pkg-config --cflags --libs opencv4`

% : %.cpp
	g++ $(CPP_FLAGS) $^ -o $@ $(LD_FLAGS)
clean:
	rm -f eye_detector
