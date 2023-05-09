# sudo apt-get install python3-dev
# sudo apt-get install libeigen3-dev
# sudo cp -r /usr/include/eigen3/Eigen /usr/local/include
# python3 -m pip install pybind11 

g++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.8 `python3 -m pybind11 --includes` jmesh_simplify.cpp -o jmesh_simplify`python3-config --extension-suffix`