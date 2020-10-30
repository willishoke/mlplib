
all:
	g++ -std=c++17 -O2 mlp.cpp -o mlp 

debug:
	g++ -std=c++17 -g mlp.cpp -o mlp 
