#Author: Forrest Iandola iandola1@illinois.edu
#Created: 1/24/10
#Modified: 9/6/12

OBJS = testTexcacheObject.o

EXENAME = testTexcacheObject

CC = nvcc 
CCOPTS = -c  -g -O3 -Xptxas -v -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35
LINK = nvcc
LINKOPTS =  -g  -o

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) $(LINKOPTS) $(EXENAME) $(OBJS)

testTexcacheObject.o : testTexcacheObject.cu
	$(CC) $(CCOPTS) testTexcacheObject.cu

clean : 
	rm -f *.o $(EXENAME) 2>/dev/null

