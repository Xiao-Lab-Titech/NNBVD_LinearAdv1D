.PHONY: run.exe

CC = g++
CPPFLAGS = -O3 -std=c++17
INCFLAGS = -I /usr/local/include/
LIBS = -L /usr/local/lib/ -lonnxruntime -fopenmp
#LIBS = -L /usr/local/lib/ -lonnxruntime.1.20.1 -fopenmp

#------------------------------------------------------------------------------ 

SRCS = main.cpp Solver.cpp Analyzer.cpp
TARGET = run.exe
OBJS = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CPPFLAGS) $(INCFLAGS) $(LIBS)

%.o: %.cpp
	$(CC) $(CPPFLAGS) $(INCFLAGS) $(LIBS) -c $<

cleanall:
	@rm -f $(OBJS) $(TARGET) *.out *.exe

clean:
	@rm -f $(OBJS) $(TARGET)

rmdata:
	@rm -f *.png *.gif *.eps *.dat

all: clean
	$(MAKE) cleanall
	$(MAKE) 


