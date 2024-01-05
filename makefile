CC = nvcc
CCFLAGS = -O3 -rdc=true -lcudadevrt -lcudart -lcurand 
LDFLAGS = 

SRCS = main.cu
INCS = color.cuh vec3.cuh ray.cuh

main: main.o
	$(CC) $(CCFLAGS) -o main main.o $(LDFLAGS) 

main.o: $(SRCS) $(INCS)
	$(CC) $(CCFLAGS) -o main.o -c main.cu $(LDFLAGS)

image.ppm: main
	rm -f image.ppm
	./main > image.ppm

clean:
	rm -f image.ppm main main.o