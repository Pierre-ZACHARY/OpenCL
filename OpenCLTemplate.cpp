//
// Created by Pierre-ZACHARY on 02/03/23.
//

#include "OpenCLTemplate.hpp"
#include <iostream>
#include <vector>
#include <fstream>

int gcd(int a, int b) {
    while (b != 0) {
        int r = a % b;
        a = b;
        b = r;
    }
    return a;
}

using namespace std;
void affiche_device(cl::Device device){
    wcout << "\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>().c_str() << endl;
    wcout << "\tDevice Version: " << device.getInfo<CL_DEVICE_VERSION >().c_str() << endl;
    wcout << "\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
    wcout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << endl;
    wcout << "\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>().c_str() << endl;
    wcout << "\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
    wcout << "\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << endl;
    wcout << "\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << endl;
    wcout << "\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << endl;
    wcout << "\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
    wcout << "\tDevice Available: " << device.getInfo<CL_DEVICE_AVAILABLE>() << endl;
    wcout << "\tMax Work-group Total Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
    vector<size_t> d= device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    wcout << "\tMax Work-group Dims: (";
    for (vector<size_t>::iterator st = d.begin(); st != d.end(); st++)
        wcout << *st << " ";
    wcout << "\x08)" << endl;
}

OpenCLTemplate::OpenCLTemplate() {
    std::vector<cl::Platform> plateformes;
    cl::Platform::get(&plateformes);
    cl::Device bestdevice;
    int max = 0;
    // find the device with CL_DEVICE_MAX_COMPUTE_UNITS max
    for (const auto & plateforme : plateformes) {
        std::vector<cl::Device> devices;
        plateforme.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (const auto & device2 : devices) {
            int computeUnits = device2.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            if (max < computeUnits) {
                bestdevice = device2;
                max = computeUnits;
            }
        }
    }
    this->device = bestdevice;
    wcout << "Device Name: " << device.getInfo<CL_DEVICE_NAME>().c_str() << endl;
//    affiche_device(bestdevice);
    this->contexte = cl::Context(device);
}

OpenCLTemplate* OpenCLTemplate::setSourceFile(std::string path, std::string kernelName) {
    std::ifstream sourceFile(path);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    this->programme = cl::Program(contexte,sourceCode);
#ifdef __WINDOWS__
    err = programme.build({device});
#else
    err = programme.build(device);
#endif
    if(err != CL_SUCCESS) {
        std::cerr << "Error on Build: " << err << std::endl;
        printf("%s", sourceCode.c_str());
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = programme.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device,&buildErr);
        std::cerr << buildInfo << std::endl << std::endl;
        exit(0);
    }

    this->kernel = cl::Kernel(programme, kernelName.c_str());
#ifdef __WINDOWS__
    cl_command_queue_properties properties = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    this->queue= cl::CommandQueue(contexte,device, &properties, &err);
#else
    this->queue= cl::CommandQueue(contexte,device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
#endif
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Error enqueuing kernel: " + std::to_string(err));
    }
    return this;
}

void OpenCLTemplate::run(cl::NDRange global, cl::NDRange local, cl::NDRange offset) {

    if(!programme())
        throw std::runtime_error("No program to run");

    try{
        err = this->queue.enqueueNDRangeKernel(kernel,offset,global,local,nullptr,&event);
        if(err != CL_SUCCESS) {
            std::cerr << "Error on Run: " << err << std::endl;
            exit(0);
        }
    } catch (...) {
        // Récupération des messages d'erreur au cas où...
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = programme.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device,&buildErr);
        std::cerr << buildInfo << std::endl << std::endl;
        exit(0);
    }
}

cl_ulong OpenCLTemplate::getExecutionTime() {
    event.wait();
    queue.finish();
    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    cl_ulong time = end - start;
    // convert time from nanoseconds to seconds
    return time;
}

void OpenCLTemplate::runAuto(int numWorkers, cl::NDRange offset) {

    const auto maxGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int numWorkersLocal;
    if(numWorkers<maxGroupSize)
        numWorkersLocal = numWorkers;
    else
        numWorkersLocal = gcd(numWorkers, maxGroupSize);
    this->run(numWorkers, numWorkersLocal, offset);

}


OpenCLTemplate::~OpenCLTemplate() = default;
