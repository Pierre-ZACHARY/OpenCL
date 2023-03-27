//
// Created by Pierre-ZACHARY on 02/03/23.
//

#ifndef OPENCL_OPENCLTEMPLATE_HPP
#define OPENCL_OPENCLTEMPLATE_HPP

#include <string>
#include <vector>
#ifdef __WINDOWS__
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif
#include <functional>

class OpenCLTemplate;

template<class T>
class OpenCLBuffer {
private:
    OpenCLTemplate *parent;
    int size;
public:
    cl::Buffer buffer;
    explicit OpenCLBuffer(OpenCLTemplate *parent, int size);
    OpenCLTemplate* write(T* ptr);
    T* read();
    OpenCLTemplate* nothing() { return parent; }
};


class OpenCLTemplate {

private :
    cl::Event event;
    cl::Device device;
    cl::Kernel kernel;
    cl::Program programme;
    int nbArgs = 0;
    std::vector<void*> buffers = std::vector<void*>();

public :
    cl::Context contexte;
    cl::CommandQueue queue;

    OpenCLTemplate();
    ~OpenCLTemplate();

    template<class T>
    OpenCLBuffer<T>* addBuffer(int size);

    template<class T>
    OpenCLTemplate* addConst(T value);

    template<class T>
    OpenCLBuffer<T>* getBuffer(int buffer_index);

    cl_ulong getExecutionTime();

    OpenCLTemplate* setSourceFile(std::string path, std::string kernelName);
    void runAuto(int numWorkers, cl::NDRange offset = cl::NullRange);
    void run(cl::NDRange global, cl::NDRange local, cl::NDRange offset = cl::NullRange);

    int err;
};

template<class T>
T *OpenCLBuffer<T>::read() {
    auto *ptr = new T[size];
    parent->queue.enqueueReadBuffer(buffer,CL_TRUE,0,size*sizeof(T), ptr);
    return ptr;
}

template<class T>
OpenCLTemplate* OpenCLBuffer<T>::write(T *ptr) {
    parent->queue.enqueueWriteBuffer(buffer , CL_TRUE, 0, size * sizeof(T) , ptr);
    return parent;
}


template<class T>
OpenCLBuffer<T>::OpenCLBuffer(OpenCLTemplate *parent, int size): parent(parent), size(size) {
    buffer = cl::Buffer(parent->contexte, CL_MEM_READ_ONLY, size * sizeof(T));
}

template<class T>
OpenCLTemplate *OpenCLTemplate::addConst(T value) {
    kernel.setArg(nbArgs, value);
    nbArgs++;
}

template<class T>
OpenCLBuffer<T> *OpenCLTemplate::addBuffer(int size) {
    if(!queue())
        throw std::runtime_error("No Queue to add buffer");
    auto * b = new OpenCLBuffer<T>(this, size);
    buffers.push_back((void*) b);
    kernel.setArg(nbArgs, b->buffer);
    nbArgs++;

    return b;
}

template<class T>
OpenCLBuffer<T> *OpenCLTemplate::getBuffer(int buffer_index) {
    if(!queue())
        throw std::runtime_error("No Queue to get buffer");
    auto* buffer = static_cast<OpenCLBuffer<T>*>(buffers[buffer_index]);
    return buffer;
}



#endif //OPENCL_OPENCLTEMPLATE_HPP
